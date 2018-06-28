package org.deeplearning4j.nn.layers.normalization;

import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.LayerHelper;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;


/**
 * Created by akshaysharma on 26/06/18.
 *
 * @author akshaysharma096
 */
public interface LayerNormalizationHelper extends LayerHelper {

    boolean checkSupported(double eps);

    Pair<Gradient, INDArray> backpropGradient(INDArray input, INDArray epsilon, int[] shape, INDArray gamma,
                                              INDArray dGammaView, INDArray dBetaView, double eps, LayerWorkspaceMgr workspaceMgr);

    INDArray preOutput(INDArray x, boolean training, int[] shape, INDArray gamma, INDArray beta, INDArray mean,
                       INDArray var, double decay, double eps, LayerWorkspaceMgr workspaceMgr);
}
