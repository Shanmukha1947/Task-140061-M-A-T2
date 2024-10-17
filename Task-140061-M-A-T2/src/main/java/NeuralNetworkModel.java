import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class NeuralNetworkModel {

    public MultiLayerNetwork init() {
        // Define the neural network configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(784).nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(128).nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(64).nOut(10)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        // Initialize the neural network model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;
    }
}

