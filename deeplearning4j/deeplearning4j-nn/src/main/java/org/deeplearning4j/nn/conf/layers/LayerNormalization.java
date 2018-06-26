package org.deeplearning4j.nn.conf.layers;

import lombok.AllArgsConstructor;
import lombok.val;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.params.LayerNormalizationParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.NoOp;

import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * Created by akshaysharma on 26/06/18.
 *
 * @author akshaysharma096
 *         Batch normalization configuration
 */

public class LayerNormalization extends FeedForwardLayer {

    protected double decay = 0.9;
    protected double eps = 1e-5;
    protected double gain = 1.0;
    protected double bias = 0.0;


    private LayerNormalization(Builder builder) {
        super(builder);
        this.decay = builder.decay;
        this.eps = builder.eps;
        this.gain = builder.gain;
        this.bias = builder.bias;
        initializeConstraints(builder);
    }

    @Override
    public LayerNormalization clone() {
        LayerNormalization clone = (LayerNormalization) super.clone();
        return clone;
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                             int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        org.deeplearning4j.nn.layers.normalization.LayerNormalization normalization_layer =
                new org.deeplearning4j.nn.layers.normalization.LayerNormalization(conf);
        normalization_layer.setListeners(trainingListeners);
        normalization_layer.setIndex(layerIndex);
        normalization_layer.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        normalization_layer.setParamTable(paramTable);
        normalization_layer.setConf(conf);
        return normalization_layer;
    }

    @Override
    public ParamInitializer initializer() {
        return LayerNormalizationParamInitializer.getInstance();
    }


    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        if (inputType == null) {
            throw new IllegalStateException(
                    "Invalid input type: Batch norm layer expected input of type CNN, got null for layer \""
                            + getLayerName() + "\"");
        }

        //Can handle CNN, flat CNN or FF input formats only
        switch (inputType.getType()) {
            case FF:
            case CNN:
            case CNNFlat:
                return inputType; //OK
            default:
                throw new IllegalStateException(
                        "Invalid input type: Batch norm layer expected input of type CNN, CNN Flat or FF, got "
                                + inputType + " for layer index " + layerIndex + ", layer name = "
                                + getLayerName());
        }
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        if (nIn <= 0 || override) {
            switch (inputType.getType()) {
                case FF:
                    nIn = ((InputType.InputTypeFeedForward) inputType).getSize();
                    break;
                case CNN:
                    nIn = ((InputType.InputTypeConvolutional) inputType).getChannels();
                    break;
                case CNNFlat:
                    nIn = ((InputType.InputTypeConvolutionalFlat) inputType).getDepth();
                default:
                    throw new IllegalStateException(
                            "Invalid input type: Layer norm layer expected input of type CNN, CNN Flat or FF, got "
                                    + inputType + " for layer " + getLayerName() + "\"");
            }
            nOut = nIn;
        }
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        if (inputType.getType() == InputType.Type.CNNFlat) {
            InputType.InputTypeConvolutionalFlat i = (InputType.InputTypeConvolutionalFlat) inputType;
            return new FeedForwardToCnnPreProcessor(i.getHeight(), i.getWidth(), i.getDepth());
        }

        return null;
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        InputType outputType = getOutputType(-1, inputType);

        //TODO CuDNN helper etc

        val numParams = initializer().numParams(this);
        int updaterStateSize = 0;

        for (String s : LayerNormalizationParamInitializer.keys()) {
            updaterStateSize += getUpdaterByParam(s).stateSize(nOut);
        }

        val inferenceWorkingSize = 2 * inputType.arrayElementsPerExample();

        val trainWorkFixed = 2 * nOut;

        val trainWorkingSizePerExample = inferenceWorkingSize //Inference during backprop
                + (outputType.arrayElementsPerExample() + 2 * nOut); //Backprop gradient calculation

        return new LayerMemoryReport.Builder(layerName, LayerNormalization.class, inputType, outputType)
                .standardMemory(numParams, updaterStateSize)
                .workingMemory(0, 0, trainWorkFixed, trainWorkingSizePerExample) //No additional memory (beyond activations) for inference
                .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching
                .build();
    }

    @Override
    public double getL1ByParam(String paramName) {
        return 0.0;
    }

    @Override
    public double getL2ByParam(String paramName) {
        return 0;
    }

    @Override
    public IUpdater getUpdaterByParam(String paramName) {
        switch (paramName) {
            case LayerNormalizationParamInitializer.BIAS:
            case LayerNormalizationParamInitializer.GAIN:
                return iUpdater;
            case LayerNormalizationParamInitializer.GLOBAL_MEAN:
            case LayerNormalizationParamInitializer.GLOBAL_VAR:
                return new NoOp();
            default:
                throw new IllegalArgumentException("Unknown parameter: \"" + paramName + "\"");
        }
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        return false; //No pretrain params in LN
    }

    @AllArgsConstructor
    public static class Builder extends FeedForwardLayer.Builder<Builder> {

        protected double decay = 0.9;
        protected double eps = 1e-5;
        protected double gain = 1.0;
        protected double bias = 0.0;
        protected List<LayerConstraint> biasConstraints;
        protected List<LayerConstraint> gainConstraints;

        public Builder(double decay) {
            this.decay = decay;
        }

        public Builder(double gain, double bias) {
            this.gain = gain;
            this.bias = bias;
        }

        public Builder() {
        }

        /**
         * Default: 1.0
         *
         * @param gain Gamma parameter for all activations, used only with locked gain/bias configuration mode
         */
        public Builder gain(double gain) {
            this.gain = gain;
            return this;
        }

        /**
         * Default: 0.0
         *
         * @param bias Beta parameter for all activations, used only with locked gamma/beta configuration mode
         */
        public Builder bias(double bias) {
            this.bias = bias;
            return this;
        }

        /**
         * Epsilon value for layer normalization; small floating point value added to variance
         *
         * @param eps Epsilon values to use
         */
        public Builder eps(double eps) {
            this.eps = eps;
            return this;
        }

        /**
         * At test time: we can use a global estimate of the mean and variance, calculated using a moving average
         * of the batch means/variances. This moving average is implemented as:<br>
         * globalMeanEstimate = decay * globalMeanEstimate + (1-decay) * batchMean<br>
         * globalVarianceEstimate = decay * globalVarianceEstimate + (1-decay) * batchVariance<br>
         *
         * @param decay Decay value to use for global stats calculation
         */
        public Builder decay(double decay) {
            this.decay = decay;
            return this;
        }

        /**
         * Set constraints to be applied to the beta parameter of this batch normalisation layer. Default: no constraints.<br>
         * Constraints can be used to enforce certain conditions (non-negativity of parameters, max-norm regularization,
         * etc). These constraints are applied at each iteration, after the parameters have been updated.
         *
         * @param constraints Constraints to apply to the beta parameter of this layer
         */
        public Builder constrainBias(LayerConstraint... constraints) {
            this.biasConstraints = Arrays.asList(constraints);
            return this;
        }

        /**
         * Set constraints to be applied to the gamma parameter of this batch normalisation layer. Default: no constraints.<br>
         * Constraints can be used to enforce certain conditions (non-negativity of parameters, max-norm regularization,
         * etc). These constraints are applied at each iteration, after the parameters have been updated.
         *
         * @param constraints Constraints to apply to the gamma parameter of this layer
         */
        public Builder constrainGain(LayerConstraint... constraints) {
            this.gainConstraints = Arrays.asList(constraints);
            return this;
        }

        Override

        public LayerNormalization build() {
            return new LayerNormalization(this);
        }

    }


}
