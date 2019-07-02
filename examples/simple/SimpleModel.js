const nn = navigator.ml.getNeuralNetworkContext();

const TENSOR_SIZE = 200;
const FLOAT_EPISILON = 1e-6;

class SimpleModel {
  constructor(arrayBuffer) {
    this.arrayBuffer_ = arrayBuffer;
    this.tensorSize_ = TENSOR_SIZE;
    this.model_ = null;
    this.compilation_ = null;
  }

  async createCompiledModel() {
    let float32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: [TENSOR_SIZE]};

    // tensor0 is a constant tensor that was established during training.
    // We read these values from the corresponding memory object.
    const tensor0 = nn.constant(float32TensorType, new Float32Array(this.arrayBuffer_, 0, TENSOR_SIZE));

    // tensor1 is one of the user provided input tensors to the trained this.model_.
    // Its value is determined pre-execution.
    const tensor1 = nn.input(float32TensorType);

    // tensor2 is a constant tensor that was established during training.
    // We read these values from the corresponding memory object.
    const tensor2 = nn.constant(float32TensorType, new Float32Array(this.arrayBuffer_, TENSOR_SIZE * Float32Array.BYTES_PER_ELEMENT, TENSOR_SIZE));

    // tensor3 is one of the user provided input tensors to the trained this.model_.
    // Its value is determined pre-execution.
    const tensor3 = nn.input(float32TensorType);

    // intermediateOutput0 is the output of the first ADD operation.
    // Its value is computed during execution.
    const intermediateOutput0 = nn.add(tensor0, tensor1);

    // intermediateOutput1 is the output of the second ADD operation.
    // Its value is computed during execution.
    const intermediateOutput1 = nn.add(tensor2, tensor3);

    // multiplierOutput is the output of the MUL operation.
    // Its value will be computed during execution.
    const multiplierOutput = nn.mul(intermediateOutput0, intermediateOutput1);

    // create a Model.
    this.model_ = await nn.createModelByOutputs([multiplierOutput], {backend: 'WASM'});

    // Create a Compilation object for the constructed this.model_.
    this.compilation_ = await this.model_.createCompilation();

    // Set the preference for the compilation, so that the runtime and drivers
    // can make better decisions.
    // Here we prefer to get the answer quickly, so we choose
    // PREFER_FAST_SINGLE_ANSWER.
    this.compilation_.setPreference(nn.PREFER_FAST_SINGLE_ANSWER);

    // Finish the compilation.
    return await this.compilation_.finish();
  }

  async compute(inputValue1, inputValue2) {
    let execution = await this.compilation_.createExecution();
    let inputTensor1 = new Float32Array(this.tensorSize_);
    inputTensor1.fill(inputValue1);
    let inputTensor2 = new Float32Array(this.tensorSize_);
    inputTensor2.fill(inputValue2);

    // Tell the execution to associate inputTensor1 to the first of the two model inputs.
    // Note that the index of the modelInput list {tensor1, tensor3}
    execution.setInput(0, inputTensor1);
    execution.setInput(1, inputTensor2);

    let outputTensor = new Float32Array(this.tensorSize_);
    execution.setOutput(0, outputTensor);

    let error = await execution.startCompute();
    if (error) {
      return error;
    }

    const goldenRef = (inputValue1 + 0.5) * (inputValue2 + 0.5);
    for (let i = 0; i < outputTensor.length; ++i) {
      let delta = Math.abs(outputTensor[i] - goldenRef);
      if (delta > FLOAT_EPISILON) {
        console.error(`Output computation error: output(${outputTensor[i]}), delta(${delta}) @ idx(${i})`)
      }
    }

    return outputTensor[0];
  }
}
