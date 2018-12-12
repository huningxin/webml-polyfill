describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Depthwise conv example-1', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op2_value = [-0.869931, 0.644628, -0.918393, 0.153672, 0.868562, -0.358177, -0.134931, -0.247565, 0.22174, -0.259157, -0.284296, -0.538065, 0.765559, 0.41986, -0.556241, 0.658494, 0.214355, -0.850169, -0.252893, -0.478935, 0.530526, -0.0700663, -0.988729, -0.303061, 0.150845, 0.829915, 0.476349, 0.406537, -0.355343, 0.757145, -0.356362, 0.800482, -0.713861, 0.210483, -0.634303, 0.718236, -0.752038, 0.457547, -0.550769, -0.551178, 0.446766, -0.227462, 0.216348, -0.852806, -0.351486, 0.55906, -0.668493, -0.303493, -0.363763, -0.162837, 0.0701012, 0.756097, -0.142269, 0.329724, -0.656317, -0.998086, -0.652949, -0.40316, -0.893682, 0.432744, 0.612362, -0.869588, -0.71327, -0.398092, -0.0423559, 0.436576, -0.925272, 0.176549, 0.822904, 0.096833, -0.296802, -0.427195, 0.031654, -0.254479, 0.244905, 0.0948254, 0.643769, -0.90391, 0.352665, -0.901179, 0.266159, -0.968068, -0.615401, -0.388975, 0.939052, -0.116289, 0.107523, -0.0582711, 0.435172, 0.334675, 0.459711, 0.717436, 0.496627, -0.680175, -0.415066, 0.339848, 0.506004, -0.337808, -0.107218, -0.172496, 0.870638, 0.931872, -0.953884, 0.903042, 0.760078, 0.209727, -0.285384, -0.45514, 0.113194, 0.0756611, 0.0924435, -0.472863, 0.960609, -0.160385, -0.839445, 0.457097, 0.163348, 0.344867, -0.131619, 0.688715, -0.540827, 0.571259, -0.95587, 0.506164, -0.155839, 0.0789621, 0.756772, -0.662069, 0.242908, 0.460821, 0.177872, -0.289839, -0.640603, 0.702598, -0.506406, -0.568262, -0.0713716, 0.413792, 0.159673, -0.305208, 0.133816, -0.160254, 0.787323, -0.753244, 0.600721, 0.263186, -0.162387, 0.477962, -0.702951, -0.731036, -0.939481, -0.524519, 0.934072, -0.511637, -0.503499, 0.106236, -0.323684, 0.534444, -0.843745, 0.364171, 0.0370358, -0.168801, -0.404559, -0.814178, 0.91745, -0.334276, 0.66925, -0.801201, 0.156511, -0.427949, 0.379153, 0.818597, -0.649902, 0.427087, -0.586015, -0.559789, -0.833923, 0.0892409, -0.621251, 0.213826, 0.465509, 0.4704, 0.380261, 0.413067, 0.180822, 0.172866, 0.59614, 0.825575, 0.662916, -0.704381, -0.297631, 0.697778];
    let op3_expect = [0.840539, -0.301347, 0.754947, -0.14848, -0.40603, 0.294432, 0.130372, 0.11573, -0.182277, 0.2504, 0.132901, 0.442306, -0.739693, -0.196274, 0.457246, -0.636246, -0.100205, 0.698864, 0.244348, 0.22389, -0.436108, 0.067699, 0.462205, 0.249125, -0.145748, -0.387964, -0.391573, -0.392801, 0.166114, -0.622396, 0.344322, -0.374205, 0.586815, -0.203372, 0.29652, -0.590411, 0.726629, -0.213891, 0.452749, 0.532555, -0.208851, 0.186981, -0.209039, 0.398664, 0.288932, -0.540171, 0.312503, 0.24948, 0.351473, 0.076122, -0.0576253, -0.73055, 0.0665069, -0.271043, 0.634142, 0.466579, 0.536743, 0.389538, 0.417773, -0.355728, -0.591672, 0.40651, 0.586329, 0.384641, 0.0198003, -0.358878, 0.894009, -0.0825318, -0.676451, -0.0935613, 0.138747, 0.351167, -0.0305845, 0.118962, -0.201319, -0.0916215, -0.300945, 0.743041, -0.34075, 0.421278, -0.218791, 0.935359, 0.287684, 0.319749, -0.907324, 0.054362, -0.0883874, 0.0563023, -0.203432, -0.275113, -0.444178, -0.335382, -0.408242, 0.657194, 0.194033, -0.279365, -0.488907, 0.157917, 0.0881365, 0.166668, -0.407001, -0.766027, 0.921655, -0.422149, -0.624807, -0.202641, 0.13341, 0.374139, -0.109369, -0.0353696, -0.0759913, 0.456887, -0.44906, 0.131841, 0.811082, -0.213681, -0.134277, -0.333215, 0.0615286, -0.566144, 0.522554, -0.267049, 0.785754, -0.489062, 0.0728509, -0.0649092, -0.731203, 0.3095, -0.199677, -0.445251, -0.0831503, 0.238257, 0.618959, -0.328446, 0.416281, 0.549062, 0.0333644, -0.340149, -0.154278, 0.142677, -0.110001, 0.15484, -0.368053, 0.619189, -0.580424, -0.123033, 0.133487, -0.461813, 0.328611, 0.600933, 0.907739, 0.245199, -0.767835, 0.49435, 0.235373, -0.0873295, 0.312748, -0.249839, 0.693584, -0.351866, -0.0173133, 0.13876, 0.39089, 0.380607, -0.754171, 0.322982, -0.312857, 0.658611, -0.151223, 0.200055, -0.311675, -0.790939, 0.303812, -0.351079, 0.566216, 0.261687, 0.68551, -0.0862257, 0.290419, -0.175771, -0.449781, -0.2199, -0.312586, -0.399111, -0.0845297, -0.142101, -0.575998, -0.385935, -0.544937, 0.680582, 0.139135, -0.573594];

    let type0 = {type: nn.INT32};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 3]};
    let type2_length = product(type2.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type1_length = product(type1.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type3_length = product(type3.dimensions);

    let b4 = operandIndex++;
    model.addOperand(type0);
    let b5 = operandIndex++;
    model.addOperand(type0);
    let b6 = operandIndex++;
    model.addOperand(type0);
    let b7 = operandIndex++;
    model.addOperand(type0);
    let b8 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let op3 = operandIndex++;
    model.addOperand(type1);
    let op0 = operandIndex++;
    model.addOperand(type2);
    let op1 = operandIndex++;
    model.addOperand(type3);

    model.setOperandValue(b4, new Int32Array([1]));
    model.setOperandValue(b5, new Int32Array([1]));
    model.setOperandValue(b6, new Int32Array([1]));
    model.setOperandValue(b7, new Int32Array([1]));
    model.setOperandValue(b8, new Int32Array([0]));
    model.setOperandValue(op0, new Float32Array([-0.966213, -0.467474, -0.82203]));
    model.setOperandValue(op1, new Float32Array([0, 0, 0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op2, op0, op1, b4, b5, b6, b7, b8], [op3]);

    model.identifyInputsAndOutputs([op2], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op2_input = new Float32Array(op2_value);
    execution.setInput(0, op2_input);

    let op3_output = new Float32Array(type1_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });

  it('check result for Depthwise conv example-2', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op2_value = [-0.295335, -0.00387601, -0.552251, 0.166084, -0.28482, -0.152143, -0.719885, -0.869386, -0.745598, 0.823947, 0.473183, -0.331337, 0.187631, 0.0426571, -0.826897, -0.755085, -0.472453, -0.0233656, 0.0483436, 0.933418, -0.961974, 0.0125783, 0.219742, 0.342604, -0.15166, 0.0934905, 0.783221, 0.129664, 0.838844, -0.271388, 0.924519, 0.342843, 0.274418, 0.350817, 0.841638, -0.543993, -0.00283395, -0.128467, -0.682943, -0.319117, 0.84634, 0.283003, 0.32865, 0.0293755, -0.0335696, 0.591266, -0.0743476, -0.741271, 0.462056, -0.583625, -0.590183, 0.6234, 0.535269, -0.670818, -0.955642, -0.770173, 0.479986, 0.664377, 0.399445, -0.968874, -0.276263, -0.901951, 0.544104, -0.958981, 0.482658, -0.807284, 0.305369, -0.947818, 0.827498, -0.382887, -0.805741, -0.796678, -0.299804, -0.229828, 0.818783, -0.103055, -0.45568, -0.227827, 0.543743, -0.96073, 0.946747, -0.857182, -0.96426, -0.292411, -0.715614, 0.765278, -0.475043, -0.590142, -0.238507, 0.673002, -0.473357, -0.319626, 0.936014, 0.486607, 0.580844, 0.425352, -0.800994, 0.290763, -0.494953, -0.441162, 0.718677, -0.828427, 0.96965, 7.53637e-05, -0.699973, -0.526886, -0.352682, 0.799466, 0.332789, 0.723389, 0.407659, -0.934084, -0.284705, 0.961484, -0.700395, -0.985808, -0.595342, -0.691721, 0.49448, -0.0842649, 0.0390966, 0.298938, -0.128094, -0.97158, 0.86393, 0.270606, -0.468986, -0.256605, 0.47215, -0.273117, -0.590343, -0.826529, -0.725381, -0.194821, -0.259661, -0.0949207, -0.180302, 0.0446834, -0.222133, -0.40393, 0.295772, -0.92949, 0.580079, -0.169856, 0.330311, 0.0173551, -0.635823, 0.475942, 0.907175, 0.242777, -0.512208, 0.362463, 0.0496289, 0.65171, 0.990057, 0.690733, -0.469013, -0.101311, -0.68372, -0.157841, -0.677711, -0.708224, -0.659437, -0.407607, 0.677033, 0.89032, 0.228307, -0.749514, 0.772958, 0.054701, 0.551705, 0.917052, -0.895022, -0.702397, 0.484142, 0.108648, 0.833347, 0.478872, -0.984112, 0.387176, -0.73299, 0.7526, 0.443312, -0.0987856, 0.125415, 0.10876, -0.498108, 0.43209, 0.344609, 0.928941, -0.130732, -0.0569167];
    let op3_expect = [0.285357, 0.00181194, 0.453967, -0.160473, 0.133146, 0.125066, 0.695562, 0.406415, 0.612903, -0.796108, -0.221201, 0.272369, -0.181291, -0.0199411, 0.679734, 0.729573, 0.22086, 0.0192072, -0.0467102, -0.436349, 0.790771, -0.0121533, -0.102724, -0.281631, 0.146536, -0.0437044, -0.643831, -0.125283, -0.392138, 0.223089, -0.893282, -0.16027, -0.22558, -0.338964, -0.393444, 0.447179, 0.0027382, 0.0600548, 0.5614, 0.308335, -0.395642, -0.232637, -0.317546, -0.0137323, 0.0275952, -0.571289, 0.0347555, 0.609347, -0.446445, 0.27283, 0.485148, -0.602337, -0.250224, 0.551432, 0.923353, 0.360036, -0.394563, -0.64193, -0.18673, 0.796443, 0.266929, 0.421638, -0.44727, 0.926579, -0.22563, 0.663612, -0.295051, 0.44308, -0.680228, 0.36995, 0.376663, 0.654893, 0.289675, 0.107439, -0.673064, 0.0995729, 0.213019, 0.18728, -0.525372, 0.449116, -0.778254, 0.82822, 0.450766, 0.24037, 0.691436, -0.357748, 0.3905, 0.570203, 0.111496, -0.553228, 0.457363, 0.149417, -0.769431, -0.470166, -0.271529, -0.349652, 0.773931, -0.135924, 0.406866, 0.426256, -0.335963, 0.680992, -0.936889, -3.52306e-05, 0.575398, 0.509084, 0.16487, -0.657185, -0.321545, -0.338165, -0.335108, 0.902524, 0.133092, -0.790369, 0.676731, 0.46084, 0.489389, 0.66835, -0.231156, 0.0692682, -0.0377757, -0.139746, 0.105297, 0.938753, -0.403865, -0.222446, 0.45314, 0.119956, -0.388121, 0.26389, 0.27597, 0.679432, 0.700873, 0.0910737, 0.213449, 0.0917136, 0.0842865, -0.0367311, 0.214628, 0.188827, -0.243133, 0.898085, -0.271172, 0.139627, -0.319151, -0.00811307, 0.522665, -0.459861, -0.424081, -0.19957, 0.494902, -0.169442, -0.0407964, -0.629691, -0.462826, -0.567803, 0.453167, 0.0473601, 0.562038, 0.152508, 0.316812, 0.582181, 0.637157, 0.190546, -0.556541, -0.860239, -0.106728, 0.616123, -0.746842, -0.0255713, -0.453518, -0.886067, 0.418399, 0.577391, -0.467784, -0.05079, -0.685036, -0.462692, 0.460047, -0.318271, 0.708224, -0.351821, -0.364416, 0.0954479, -0.0586282, -0.0894044, 0.481278, -0.201991, -0.283279, -0.897555, 0.0611137, 0.0467872];

    let type0 = {type: nn.INT32};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 3]};
    let type2_length = product(type2.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type1_length = product(type1.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type3_length = product(type3.dimensions);

    let b4 = operandIndex++;
    model.addOperand(type0);
    let b5 = operandIndex++;
    model.addOperand(type0);
    let b6 = operandIndex++;
    model.addOperand(type0);
    let b7 = operandIndex++;
    model.addOperand(type0);
    let b8 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let op3 = operandIndex++;
    model.addOperand(type1);
    let op0 = operandIndex++;
    model.addOperand(type2);
    let op1 = operandIndex++;
    model.addOperand(type3);

    model.setOperandValue(b4, new Int32Array([1]));
    model.setOperandValue(b5, new Int32Array([1]));
    model.setOperandValue(b6, new Int32Array([1]));
    model.setOperandValue(b7, new Int32Array([1]));
    model.setOperandValue(b8, new Int32Array([0]));
    model.setOperandValue(op0, new Float32Array([-0.966213, -0.467474, -0.82203]));
    model.setOperandValue(op1, new Float32Array([0, 0, 0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op2, op0, op1, b4, b5, b6, b7, b8], [op3]);

    model.identifyInputsAndOutputs([op2], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op2_input = new Float32Array(op2_value);
    execution.setInput(0, op2_input);

    let op3_output = new Float32Array(type1_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });
});
