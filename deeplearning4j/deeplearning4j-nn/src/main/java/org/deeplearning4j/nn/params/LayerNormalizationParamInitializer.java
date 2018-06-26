package org.deeplearning4j.nn.params;

import lombok.val;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.LayerNormalization;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.*;

/**
 * Created by akshaysharma on 26/06/18.
 *
 * @author akshaysharma096
 */
public class LayerNormalizationParamInitializer implements ParamInitializer {

    private static final LayerNormalizationParamInitializer INSTANCE = new LayerNormalizationParamInitializer();

    public static LayerNormalizationParamInitializer getInstance() {
        return INSTANCE;
    }

    public static final String GAIN = "gain";
    public static final String BIAS = "bias";
    public static final String GLOBAL_MEAN = "mean";
    public static final String GLOBAL_VAR = "var";

    public static List<String> keys() {
        return Arrays.asList(GAIN, BIAS, GLOBAL_MEAN, GLOBAL_VAR);
    }

    @Override
    public long numParams(NeuralNetConfiguration conf) {
        return numParams(conf.getLayer());
    }

    @Override
    public long numParams(Layer l) {
        LayerNormalization layer = (LayerNormalization) l;
        //Parameters in layer norm:
        //gain, bias, global mean estimate, global variance estimate
        return 4 * layer.getNOut();
    }

    @Override
    public List<String> paramKeys(Layer layer) {
        return Arrays.asList(GAIN, BIAS, GLOBAL_MEAN, GLOBAL_VAR);
    }

    @Override
    public List<String> weightKeys(Layer layer) {
        return Collections.emptyList();
    }

    @Override
    public List<String> biasKeys(Layer layer) {
        return Collections.emptyList();
    }

    @Override
    public boolean isWeightParam(Layer layer, String key) {
        return false;
    }

    @Override
    public boolean isBiasParam(Layer layer, String key) {
        return true;
    }

    @Override
    public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
        Map<String, INDArray> params = Collections.synchronizedMap(new LinkedHashMap<String, INDArray>());
        LayerNormalization layer = (LayerNormalization) conf.getLayer();

        long nOut = layer.getNOut();
        long meanOffset = 0;
        INDArray gammaView = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nOut));
        INDArray betaView = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(nOut, 2 * nOut));

        params.put(GAIN, createGain(conf, gammaView, initializeParams));
        conf.addVariable(GAIN);
        params.put(BIAS, createBias(conf, betaView, initializeParams));
        conf.addVariable(BIAS);
        meanOffset = 2 * nOut;

        INDArray globalMeanView =
                paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(meanOffset, meanOffset + nOut));
        INDArray globalVarView = paramsView.get(NDArrayIndex.point(0),
                NDArrayIndex.interval(meanOffset + nOut, meanOffset + 2 * nOut));

        if (initializeParams) {
            globalMeanView.assign(0);
            globalVarView.assign(1);
        }

        params.put(GLOBAL_MEAN, globalMeanView);
        conf.addVariable(GLOBAL_MEAN);
        params.put(GLOBAL_VAR, globalVarView);
        conf.addVariable(GLOBAL_VAR);
        return params;
    }

    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
        LayerNormalization layer = (LayerNormalization) conf.getLayer();
        long nOut = layer.getNOut();
        Map<String, INDArray> out = new LinkedHashMap<String, INDArray>();
        long meanOffset = 0;
        INDArray gainView = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nOut));
        INDArray biasView = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(nOut, 2 * nOut));
        out.put(GAIN, gainView);
        out.put(BIAS, biasView);
        meanOffset = 2 * nOut;

        out.put(GLOBAL_MEAN,
                gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(meanOffset, meanOffset + nOut)));
        out.put(GLOBAL_VAR, gradientView.get(NDArrayIndex.point(0),
                NDArrayIndex.interval(meanOffset + nOut, meanOffset + 2 * nOut)));
        return out;
    }

    private INDArray createGain(NeuralNetConfiguration conf, INDArray gainView, boolean initializeParams) {
        LayerNormalization layer = (LayerNormalization) conf.getLayer();
        if (initializeParams)
            gainView.assign(layer.getGain());
        return gainView;
    }

    private INDArray createBias(NeuralNetConfiguration conf, INDArray biasView, boolean initializeParams) {
        LayerNormalization layer = (LayerNormalization) conf.getLayer();
        if (initializeParams)
            biasView.assign(layer.getBias());
        return biasView;
    }
}
