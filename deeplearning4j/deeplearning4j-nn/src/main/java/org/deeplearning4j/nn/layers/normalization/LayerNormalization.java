package org.deeplearning4j.nn.layers.normalization;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.layers.LayerHelper;
import org.deeplearning4j.nn.params.LayerNormalizationParamInitializer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAddOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastDivOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastSubOp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.util.OneTimeLogger;

import java.util.*;


/**
 * Created by akshaysharma on 26/06/18.
 *
 * @author akshaysharma096
 */

/**
 * Layer normalization layer.
 * References:
 * https://arxiv.org/pdf/1607.06450.pdf
 * http://mlexplained.com/2018/01/13/weight-normalization-and-layer-normalization-explained-normalization-in-deep-learning-part-2
 **/

@Slf4j
public class LayerNormalization extends BaseLayer<org.deeplearning4j.nn.conf.layers.LayerNormalization> {

    LayerNormalizationHelper helper = null;
    protected int index = 0;
    protected List<TrainingListener> listeners = new ArrayList<>();
    protected INDArray std;
    protected INDArray xMu;
    protected INDArray xHat;

    public LayerNormalization(NeuralNetConfiguration conf) {
        super(conf);
        initializeHelper();
    }

    void initializeHelper() {
        String backend = Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend");
        if ("CUDA".equalsIgnoreCase(backend)) {
            try {
                helper = Class.forName("org.deeplearning4j.nn.layers.normalization.CudnnLayerNormalizationHelper")
                        .asSubclass(LayerNormalizationHelper.class).newInstance();
                log.debug("CudnnLayerNormalizationHelper successfully initialized");
                if (!helper.checkSupported(layerConf().getEps())) {
                    helper = null;
                }
            } catch (Throwable t) {
                if (!(t instanceof ClassNotFoundException)) {
                    log.warn("Could not initialize CudnnLayerNormalizationHelper", t);
                } else {
                    OneTimeLogger.info(log, "cuDNN not found: "
                            + "use cuDNN for better GPU performance by including the deeplearning4j-cuda module. "
                            + "For more information, please refer to: https://deeplearning4j.org/cudnn", t);
                }
            }
        }
    }

    @Override
    public double calcL2(boolean backpropParamsOnly) {
        return 0;
    }

    @Override
    public double calcL1(boolean backpropParamsOnly) {
        return 0;
    }

    @Override
    public Type type() {
        return Type.NORMALIZATION;
    }

    @Override
    public INDArray preOutput(INDArray x, TrainingMode training, LayerWorkspaceMgr workspaceMgr) {
        INDArray activations;

        org.deeplearning4j.nn.conf.layers.LayerNormalization layerConf = layerConf();
        val shape = getShape(x);


        INDArray gain = null;
        INDArray bias = null;
        INDArray globalMeanView = getParam(LayerNormalizationParamInitializer.GLOBAL_MEAN);
        INDArray globalVarView = getParam(LayerNormalizationParamInitializer.GLOBAL_VAR);

        gain = getParam(LayerNormalizationParamInitializer.GAIN);
        bias = getParam(LayerNormalizationParamInitializer.BIAS);

        if (helper != null && input.rank() == 4) {
            //Note that cudnn does not support dense (2d) batch norm case as of v7.1
            double decay = layerConf.getDecay();

            // FIXME: int cast
            INDArray ret = helper.preOutput(x, training == TrainingMode.TRAIN, ArrayUtil.toInts(shape), gain, bias, globalMeanView,
                    globalVarView, decay, layerConf.getEps(), workspaceMgr);
            if (ret != null) {
                return ret;
            }
        }

        INDArray mean, var;
        if (training == TrainingMode.TRAIN) {
            switch (x.rank()) {
                case 2:
                    // mean and variance over a single training case
                    mean = x.mean(1);
                    var = x.var(false, 1);
                    break;
                case 4:
                    // mean and variance over samples AND locations
                    mean = x.mean(0, 2, 3);
                    var = x.var(false, 0, 2, 3);
                    break;
                default:
                    throw new IllegalStateException("Layer normalization on activations of rank " + x.rank()
                            + " not supported " + layerId());
            }

            var.addi(layerConf.getEps());
        } else {
            // Global mean and variance estimate - used after training
            mean = getParam(LayerNormalizationParamInitializer.GLOBAL_MEAN);
            var = getParam(LayerNormalizationParamInitializer.GLOBAL_VAR);
        }

        std = Transforms.sqrt(workspaceMgr.dup(ArrayType.INPUT, var), false);

        if (x.rank() == 2) {
            xMu = workspaceMgr.leverageTo(ArrayType.INPUT, x.subRowVector(mean));
            xHat = workspaceMgr.leverageTo(ArrayType.INPUT, xMu.divRowVector(std));

            activations = xHat.mulRowVector(gain).addiRowVector(bias);

        } else if (x.rank() == 4) {
            if (!Shape.strideDescendingCAscendingF(x))
                x = x.dup(); //TODO: temp Workaround for broadcast bug. To be removed when fixed
            xMu = workspaceMgr.createUninitialized(ArrayType.INPUT, x.shape(), x.ordering());
            xMu = Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(x, mean, xMu, 1));
            xHat = workspaceMgr.createUninitialized(ArrayType.INPUT, x.shape(), x.ordering());
            xHat = Nd4j.getExecutioner().execAndReturn(new BroadcastDivOp(xMu, std, xHat, 1));


            //Standard case: gamma and beta are learned per parameter
            activations = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, x.shape(), x.ordering());
            activations = Nd4j.getExecutioner().execAndReturn(
                    new BroadcastMulOp(xHat, gain, activations, 1));
            activations = Nd4j.getExecutioner()
                    .execAndReturn(new BroadcastAddOp(activations, bias, activations, 1));
        } else {
            throw new IllegalStateException(
                    "The layer prior to BatchNorm in the configuration is not currently supported. "
                            + layerId());
        }
        if (training == TrainingMode.TRAIN) {
            globalMeanView.assign(mean);
            globalVarView.assign(var);
        }

        activations = workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, activations);   //Most of the time this should be a no-op
        return activations;
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);
        INDArray nextEpsilon;
        val shape = getShape(epsilon);
        val batchSize = epsilon.size(1); // number of activation
        org.deeplearning4j.nn.conf.layers.LayerNormalization layerConf = layerConf();
        INDArray gain = null;
        INDArray dGainView;
        INDArray dBiasView;
        INDArray dGlobalMeanView = gradientViews.get(LayerNormalizationParamInitializer.GLOBAL_MEAN);
        INDArray dGlobalVarView = gradientViews.get(LayerNormalizationParamInitializer.GLOBAL_VAR);
        gain = getParam(LayerNormalizationParamInitializer.GAIN);
        dGainView = gradientViews.get(LayerNormalizationParamInitializer.GAIN);
        dBiasView = gradientViews.get(LayerNormalizationParamInitializer.BIAS);

        Gradient retGradient = new DefaultGradient();

        if (helper != null && epsilon.rank() == 4) {
            Pair<Gradient, INDArray> ret = helper.backpropGradient(input, epsilon, ArrayUtil.toInts(shape), gain, dGainView, dBiasView,
                    layerConf.getEps(), workspaceMgr);
            if (ret != null) {
                ret.getFirst().setGradientFor(LayerNormalizationParamInitializer.GLOBAL_MEAN, dGlobalMeanView);
                ret.getFirst().setGradientFor(LayerNormalizationParamInitializer.GLOBAL_VAR, dGlobalVarView);
                return ret;
            }
        }

        if (epsilon.rank() == 2) {
            INDArray dBias = epsilon.sum(0);
            INDArray dGain = epsilon.mul(xHat).sum(0);
            INDArray dxhat;
            dxhat = epsilon.mulRowVector(gain);

            INDArray dLdVar = dxhat.mul(xMu).sum(0).muli(-0.5).muli(Transforms.pow(std, -3.0, true));

            //dL/dmu
            INDArray dxmu1 = dxhat.sum(0).divi(std).negi();
            INDArray dxmu2 = xMu.sum(0).muli(-2.0 / batchSize).muli(dLdVar);

            INDArray dLdmu = dxmu1.addi(dxmu2);

            INDArray dLdx = dxhat.diviRowVector(std).addi(xMu.muliRowVector(dLdVar.muli(2.0 / batchSize)))
                    .addiRowVector(dLdmu.muli(1.0 / batchSize));


            dGainView.assign(dGain);
            dBiasView.assign(dBias);

            retGradient.setGradientFor(LayerNormalizationParamInitializer.GAIN, dGainView);
            retGradient.setGradientFor(LayerNormalizationParamInitializer.BIAS, dBiasView);
            dGlobalMeanView.assign(0);
            dGlobalVarView.assign(0);
            retGradient.setGradientFor(LayerNormalizationParamInitializer.GLOBAL_MEAN, dGlobalMeanView);
            retGradient.setGradientFor(LayerNormalizationParamInitializer.GLOBAL_VAR, dGlobalVarView);

            nextEpsilon = dLdx;

        } else if (epsilon.rank() == 4) {
            INDArray dBias = epsilon.sum(0, 2, 3);
            INDArray dGain = epsilon.mul(xHat).sum(0, 2, 3);
            INDArray dxhat;

            dxhat = Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(epsilon, gain,
                    Nd4j.createUninitialized(epsilon.shape(), epsilon.ordering()), 1));


            INDArray dLdVar = dxhat.mul(xMu).sum(0, 2, 3).muli(-0.5).muli(Transforms.pow(std, -3.0, true));


            val effectiveBatchSize = input.size(0) * input.size(2) * input.size(3);
            INDArray dxmu1 = dxhat.sum(0, 2, 3).divi(std).negi();
            INDArray dxmu2 = xMu.sum(0, 2, 3).muli(-2.0 / effectiveBatchSize).muli(dLdVar);
            INDArray dLdmu = dxmu1.addi(dxmu2);

            INDArray dLdx = Nd4j.getExecutioner().execAndReturn(new BroadcastDivOp(dxhat, std, dxhat, 1))
                    .addi(Nd4j.getExecutioner().execAndReturn(
                            new BroadcastMulOp(xMu, dLdVar.muli(2.0 / effectiveBatchSize), xMu, 1)));
            Nd4j.getExecutioner()
                    .execAndReturn(new BroadcastAddOp(dLdx, dLdmu.muli(1.0 / effectiveBatchSize), dLdx, 1));

            dGainView.assign(dGain);
            dBiasView.assign(dBias);

            retGradient.setGradientFor(LayerNormalizationParamInitializer.GAIN, dGainView);
            retGradient.setGradientFor(LayerNormalizationParamInitializer.BIAS, dBiasView);

            dGlobalMeanView.assign(0);
            dGlobalVarView.assign(0);
            retGradient.setGradientFor(LayerNormalizationParamInitializer.GLOBAL_MEAN, dGlobalMeanView);
            retGradient.setGradientFor(LayerNormalizationParamInitializer.GLOBAL_VAR, dGlobalVarView);

            nextEpsilon = dLdx;
        } else {
            throw new IllegalStateException(
                    "The layer prior to LayerNorm in the configuration is not currently supported. "
                            + layerId());
        }

        //TODO could optimize this
        nextEpsilon = workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, nextEpsilon);
        return new Pair<>(retGradient, nextEpsilon);

    }

    @Override
    public void fit(INDArray input, LayerWorkspaceMgr workspaceMgr) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        return preOutput(input, training ? TrainingMode.TRAIN : TrainingMode.TEST, workspaceMgr);
    }

    @Override
    public Layer transpose() {
        throw new UnsupportedOperationException(layerId());

    }

    @Override
    public Layer clone() {
        throw new UnsupportedOperationException(layerId());

    }

    @Override
    public Collection<TrainingListener> getListeners() {
        return listeners;
    }

    @Override
    public void setListeners(TrainingListener... listeners) {
        this.listeners = new ArrayList<>(Arrays.asList(listeners));
    }

    @Override
    public void setIndex(int index) {
        this.index = index;
    }

    @Override
    public int getIndex() {
        return index;
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public LayerHelper getHelper() {
        return helper;
    }

    public long[] getShape(INDArray x) {
        if (x.rank() == 2 || x.rank() == 4)
            return new long[]{1, x.size(1)};
        if (x.rank() == 3) {
            val wDim = x.size(1);
            val hdim = x.size(2);
            if (x.size(0) > 1 && wDim * hdim == x.length())
                throw new IllegalArgumentException("Illegal input for batch size " + layerId());
            return new long[]{1, wDim * hdim};
        } else
            throw new IllegalStateException("Unable to process input of rank " + x.rank() + " " + layerId());
    }
}
