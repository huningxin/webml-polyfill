function product(shape) {
  let result = 1;
  for (let i = 0; i < shape.length; i++) {
    result = result * shape[i];
  }
  return result;
}

async function createWebNNConv(hasBias, hasRelu) {
  const nn = navigator.ml.getNeuralNetworkContext();
  options={
    "backend": "WebML",
    "prefer": "sustained"
  };
  let model = await nn.createModel(options);
  let operandIndex = 0;

  // inputDims [n,h,w,i]
  // filterDims [h,w,i,o]
  const inputDesc = {type: nn.TENSOR_FLOAT32, dimensions: inputDims};
  const filterDesc = {type: nn.TENSOR_FLOAT32, dimensions: [filterDims[3], filterDims[0], filterDims[1], filterDims[2]]};
  const biasDesc = {type: nn.TENSOR_FLOAT32, dimensions: [filterDims[3]]};
  const intDesc = {type: nn.INT32};

  const input = operandIndex++;
  model.addOperand(inputDesc);
  const filter = operandIndex++;
  model.addOperand(filterDesc);
  const bias = operandIndex++;
  model.addOperand(biasDesc);
  const pad = operandIndex++;
  model.addOperand(intDesc);
  const act = operandIndex++;
  model.addOperand(intDesc);
  const stride = operandIndex++;
  model.addOperand(intDesc);
  const output = operandIndex++;
  model.addOperand(inputDesc);

  const filterData = await tf.ones(filterDims).data();
  const biasData = hasBias ? await tf.ones([filterDims[3]]).data() : await tf.zeros([filterDims[3]]).data()
  model.setOperandValue(filter, filterData);
  model.setOperandValue(bias, biasData);
  model.setOperandValue(pad, new Int32Array([(filterDims[1]-1)/2]));
  model.setOperandValue(act, new Int32Array([hasRelu?nn.FUSE_RELU:nn.FUSE_NONE]));
  model.setOperandValue(stride, new Int32Array([1]));
  model.addOperation(nn.CONV_2D, [input, filter, bias, pad, pad, pad, pad, stride, stride, act], [output]);

  model.identifyInputsAndOutputs([input], [output]);
  await model.finish();
  return model;
}

function executeWebNNForGPU(device, execution, input, output) {
  const commandEncoder = device.createCommandEncoder();
  commandEncoder.setNnGraphInput(input, 0, execution);
  commandEncoder.setNnGraphOutput(output, 0, execution);
  commandEncoder.executeNnGraph(execution);
  device.getQueue().submit([commandEncoder.finish()]);
}

const iterations = 100;

async function tfConv2d(inputDims,filterDims){
  await tf.ready();
  tf.setBackend('webgpu');
  await tf.ready();
  const input = tf.ones(inputDims);
  const filter = tf.ones(filterDims);
  const bias = tf.ones([filterDims[3]]);
  //warm up
  let convOutput = tf.conv2d(input, filter, 1, 'same');
  let addOutput = tf.add(convOutput, bias);
  let reluOutput = tf.relu(addOutput);
  let result = await reluOutput.data();
  let start = performance.now();
  for(let i=0; i<iterations; i++){
    convOutput = tf.conv2d(input, filter, 1, 'same');
    addOutput = tf.add(convOutput, bias);
    reluOutput = tf.relu(addOutput);
    result = await reluOutput.data();
  }
  let elapsedTime = ((performance.now() - start) / iterations).toFixed(2);
  document.getElementById('output').innerHTML += `WebGPU conv2d/add/relu elapsed time: ${elapsedTime} ms <br/>`;
  console.log(result);
}

async function WebNNConvGPUWithTf(inputDims,filterDims) {
  await tf.ready();
  tf.setBackend('webgpu');
  await tf.ready();
  let device=tf.backend().device;
  const nn = navigator.ml.getNeuralNetworkContext();
  let nnConv = await createWebNNConv(false, false);
  let compilation = await nnConv.createCompilation();
  compilation.setPreference(nn.PREFER_SUSTAINED_SPEED);
  await compilation.finish();
  let execution = await compilation.createExecution();

  const inputTensor = tf.ones(inputDims);
  const outputTensor = tf.zeros(inputDims);

  let inputBuffer = await tf.backend().getGPUBuffer(inputTensor.dataId);
  let outputBuffer = await tf.backend().getGPUBuffer(outputTensor.dataId);

  executeWebNNForGPU(device, execution, inputBuffer, outputBuffer);
  const biasTensor = tf.ones([filterDims[3]]);
  let addOutput = tf.add(outputTensor, biasTensor);
  let reluOutput = tf.relu(addOutput);
  let result = await reluOutput.data();
  let start = performance.now();
  for (let i=0; i<iterations; i++) {
    executeWebNNForGPU(device, execution, inputBuffer, outputBuffer);
    addOutput = tf.add(outputTensor, biasTensor);
    reluOutput = tf.relu(addOutput);
    result = await reluOutput.data();
  }
  const  elapsedTime =((performance.now() - start) / iterations).toFixed(2);
  document.getElementById('output').innerHTML += `WebNN conv2d interops with WebGPU add/relu via WebGPUBuffer elapsed time: ${elapsedTime} ms <br/>`;
  console.log(result);
}

async function WebNNConvGPU(inputDims,filterDims) {
  await tf.ready();
  tf.setBackend('webgpu');
  await tf.ready();
  let device = tf.backend().device;
  const nn = navigator.ml.getNeuralNetworkContext();
  let nnConv = await createWebNNConv(true, true);
  let compilation = await nnConv.createCompilation();
  compilation.setPreference(nn.PREFER_SUSTAINED_SPEED);
  await compilation.finish();
  let execution = await compilation.createExecution();

  const inputTensor = tf.ones(inputDims);
  const outputTensor = tf.zeros(inputDims);

  let inputBuffer = await tf.backend().getGPUBuffer(inputTensor.dataId);
  let outputBuffer = await tf.backend().getGPUBuffer(outputTensor.dataId);

  let start = performance.now();
  let result;
  for (let i = 0; i < iterations; i++) {
    executeWebNNForGPU(device, execution, inputBuffer, outputBuffer);
    result = await outputTensor.data();
  }
  const elapsedTime =((performance.now() - start) / iterations).toFixed(2);
  document.getElementById('output').innerHTML += `WebNN conv2d with fused bias/relu elapsed time: ${elapsedTime} ms <br/>`;
  console.log(result);  
}

async function WebNNConvCPU(inputDims,filterDims) {
  await tf.ready();
  tf.setBackend('webgpu');
  await tf.ready();
  let device=tf.backend().device;
  const nn = navigator.ml.getNeuralNetworkContext();
  let model = await createWebNNConv(true, true);
  let compilation = await model.createCompilation();
  compilation.setPreference(nn.PREFER_SUSTAINED_SPEED);
  await compilation.finish();
  let execution = await compilation.createExecution();

  const input = await tf.ones(inputDims).data();
  const output = await tf.zeros(inputDims).data();

  execution.setInput(0, input);
  execution.setOutput(0, output);

  let start = performance.now();
  for (let i=0; i < iterations; i++) {
    await execution.startCompute();
  }
  const elapsedTime =((performance.now() - start) / iterations).toFixed(2);
  document.getElementById('output').innerHTML = `WebNN conv2d with fused bias /relu via ArrayBuffer elapsed time: ${elapsedTime} ms <br/>`;
  console.log(output);
}

async function WebNNConvCPUWithTf(inputDims,filterDims) {
  await tf.ready();
  tf.setBackend('webgpu');
  await tf.ready();
  let device=tf.backend().device;
  const nn = navigator.ml.getNeuralNetworkContext();
  let model = await createWebNNConv(false, false);
  let compilation = await model.createCompilation();
  compilation.setPreference(nn.PREFER_SUSTAINED_SPEED);
  await compilation.finish();
  let execution = await compilation.createExecution();

  const input = await tf.ones(inputDims).data();
  const output = await tf.zeros(inputDims).data();

  execution.setInput(0, input);
  execution.setOutput(0, output);
  await execution.startCompute();
  const biasTensor = tf.ones([filterDims[3]]);
  let outputTensor = tf.tensor(output, inputDims);
  let addOutput = tf.add(outputTensor, biasTensor);
  let reluOutput = tf.relu(addOutput);
  let result = await reluOutput.data();
  let start = performance.now();
  for (let i=0; i < iterations; i++) {
    await execution.startCompute();
    outputTensor = tf.tensor(output, inputDims);
    addOutput = tf.add(outputTensor, biasTensor);
    reluOutput = tf.relu(addOutput);
    result = await reluOutput.data();
  }
  const elapsedTime =((performance.now() - start) / iterations).toFixed(2);
  document.getElementById('output').innerHTML += `WebNN conv2d interops with WebGPU add/relu via ArrayBuffer elapsed time: ${elapsedTime} ms <br/>`;
  console.log(result);
}


async function tfConv2dx2(inputDims,filterDims){
  await tf.ready();
  tf.setBackend('webgpu');
  await tf.ready();
  const inputData = new Float32Array(product(inputDims));
  inputData.fill(0.1);
  const input = tf.tensor(inputData, inputDims);
  const filter = tf.ones(filterDims);
  const bias = tf.ones([filterDims[3]]);
  //warm up
  let im0 = tf.conv2d(input, filter, 1, 'same');
  let im1 = tf.add(im0, bias);
  let im2 = tf.relu(im1);
  let im3 = tf.conv2d(im2, filter, 1, 'same');
  let im4 = tf.add(im3, bias);
  let im5 = tf.relu(im4);
  let result = await im5.data();
  let start = performance.now();
  for(let i=0; i<iterations; i++){
    im0 = tf.conv2d(input, filter, 1, 'same');
    im1 = tf.add(im0, bias);
    im2 = tf.relu(im1);
    im3 = tf.conv2d(im2, filter, 1, 'same');
    im4 = tf.add(im3, bias);
    im5 = tf.relu(im4);
    result = await im5.data();
  }
  console.log(result);
  let elapsedTime = ((performance.now() - start) / iterations).toFixed(2);
  document.getElementById('output').innerHTML += `tfConv2d x2 elapsed time: ${elapsedTime} ms <br/>`;
}


async function WebNNConvGPUx2(inputDims,filterDims) {
  await tf.ready();
  tf.setBackend('webgpu');
  await tf.ready();
  let device=tf.backend().device;
  const nn = navigator.ml.getNeuralNetworkContext();
  options={
    "backend": "WebML",
    "prefer": "sustained"
  };
  let model = await nn.createModel(options);
  let operandIndex = 0;

  // inputDims [n,h,w,i]
  // filterDims [h,w,i,o]
  const inputDesc = {type: nn.TENSOR_FLOAT32, dimensions: inputDims};
  const filterDesc = {type: nn.TENSOR_FLOAT32, dimensions: [filterDims[3], filterDims[0], filterDims[1], filterDims[2]]};
  const biasDesc = {type: nn.TENSOR_FLOAT32, dimensions: [filterDims[3]]};
  const intDesc = {type: nn.INT32};

  const input = operandIndex++;
  model.addOperand(inputDesc);
  const filter = operandIndex++;
  model.addOperand(filterDesc);
  const bias = operandIndex++;
  model.addOperand(biasDesc);
  const pad = operandIndex++;
  model.addOperand(intDesc);
  const act = operandIndex++;
  model.addOperand(intDesc);
  const stride = operandIndex++;
  model.addOperand(intDesc);
  const immediateOutput = operandIndex++;
  model.addOperand(inputDesc);
  const output = operandIndex++;
  model.addOperand(inputDesc);

  const filterData = await tf.ones(filterDims).data();
  const biasData = await tf.ones([filterDims[3]]).data();
  model.setOperandValue(filter, filterData);
  model.setOperandValue(bias, biasData);
  model.setOperandValue(pad, new Int32Array([(filterDims[1]-1)/2]));
  model.setOperandValue(act, new Int32Array([nn.FUSE_RELU]));
  model.setOperandValue(stride, new Int32Array([1]));
  model.addOperation(nn.CONV_2D, [input, filter, bias, pad, pad, pad, pad, stride, stride, act], [immediateOutput]);
  model.addOperation(nn.CONV_2D, [immediateOutput, filter, bias, pad, pad, pad, pad, stride, stride, act], [output]);

  model.identifyInputsAndOutputs([input], [output]);
  await model.finish();
  let compilation = await model.createCompilation();
  compilation.setPreference(nn.PREFER_SUSTAINED_SPEED);
  await compilation.finish();
  let execution = await compilation.createExecution();

  const inputData = new Float32Array(product(inputDims));
  inputData.fill(0.1);
  const inputTensor = tf.tensor(inputData, inputDims);
  const outputTensor = tf.zeros(inputDims);

  let inputBuffer = await tf.backend().getGPUBuffer(inputTensor.dataId);
  let outputBuffer = await tf.backend().getGPUBuffer(outputTensor.dataId);

  let result;
  let start = performance.now();
  for (let i=0;i<iterations;i++) {
    executeWebNNForGPU(device, execution, inputBuffer, outputBuffer);
    result = await outputTensor.data();
  }
  console.log(result);

  const  elapsedTime =((performance.now() - start) / iterations).toFixed(2);
  document.getElementById('output').innerHTML = `WebNN conv x2 elapsed time: ${elapsedTime} ms <br/>`;
}

async function WebNNConvGPUx2Model(inputDims,filterDims) {
  await tf.ready();
  tf.setBackend('webgpu');
  await tf.ready();
  let device=tf.backend().device;
  const nn = navigator.ml.getNeuralNetworkContext();

  let model1 = await createWebNNConv(true, true);
  let compilation1 = await model1.createCompilation();
  compilation1.setPreference(nn.PREFER_SUSTAINED_SPEED);
  await compilation1.finish();
  let execution1 = await compilation1.createExecution();

  let model2 = await createWebNNConv(true, true);
  let compilation2 = await model2.createCompilation();
  compilation2.setPreference(nn.PREFER_SUSTAINED_SPEED);
  await compilation2.finish();
  let execution2 = await compilation2.createExecution();

  const inputData = new Float32Array(product(inputDims));
  inputData.fill(0.1);
  const inputTensor = tf.tensor(inputData, inputDims);
  const immediateTensor = tf.zeros(inputDims);
  const outputTensor = tf.zeros(inputDims);

  let inputBuffer = await tf.backend().getGPUBuffer(inputTensor.dataId);
  let immediateBuffer = await tf.backend().getGPUBuffer(immediateTensor.dataId);
  let outputBuffer = await tf.backend().getGPUBuffer(outputTensor.dataId);

  let result;
  let start = performance.now();
  for (let i=0;i<iterations;i++) {
    // workaround: use integer to index graph
    const commandEncoder = device.createCommandEncoder();
    commandEncoder.setNnGraphInput(inputBuffer, 0, 0);
    commandEncoder.setNnGraphOutput(immediateBuffer, 0, 0);
    commandEncoder.executeNnGraph(0);
    commandEncoder.setNnGraphInput(immediateBuffer, 0, 1);
    commandEncoder.setNnGraphOutput(outputBuffer, 0, 1);
    commandEncoder.executeNnGraph(1);
    device.getQueue().submit([commandEncoder.finish()]);
    result = await outputTensor.data();
  }
  console.log(result);

  const  elapsedTime =((performance.now() - start) / iterations).toFixed(2);
  document.getElementById('output').innerHTML += `WebNN conv x2 ops elapsed time: ${elapsedTime} ms <br/>`;
}

async function WebNNConvGPUx2WithTf(inputDims,filterDims) {
  await tf.ready();
  tf.setBackend('webgpu');
  await tf.ready();
  let device=tf.backend().device;
  const nn = navigator.ml.getNeuralNetworkContext();
  let model = await createWebNNConv(false, false);
  let compilation = await model.createCompilation();
  compilation.setPreference(nn.PREFER_SUSTAINED_SPEED);
  await compilation.finish();
  let execution = await compilation.createExecution();

  const inputData = new Float32Array(product(inputDims));
  inputData.fill(0.1);
  const inputTensor = tf.tensor(inputData, inputDims);
  const outputTensor = tf.zeros(inputDims);

  let inputBuffer = await tf.backend().getGPUBuffer(inputTensor.dataId);
  let outputBuffer = await tf.backend().getGPUBuffer(outputTensor.dataId);

  const filterTensor = tf.ones(filterDims);
  const biasTensor = tf.ones([filterDims[3]]);
  executeWebNNForGPU(device, execution, inputBuffer, outputBuffer);
  let convOutput = tf.conv2d(outputTensor, filterTensor, 1, 'same');
  let addOutput = tf.add(convOutput, biasTensor);
  let reluOutput = tf.relu(addOutput);
  let result = await reluOutput.data();
  let start = performance.now();
  for (let i = 0; i < iterations; i++) {
    executeWebNNForGPU(device, execution, inputBuffer, outputBuffer);
    convOutput = tf.conv2d(outputTensor, filterTensor, 1, 'same');
    addOutput = tf.add(convOutput, biasTensor);
    reluOutput = tf.relu(addOutput);
    result = await reluOutput.data();
  }
  console.log(result);

  const  elapsedTime =((performance.now() - start) / iterations).toFixed(2);
  document.getElementById('output').innerHTML = `WebNN conv + tf.conv2d elapsed time: ${elapsedTime} ms <br/>`;
}

const inputDims = [1, 100, 100, 100];
const filterDims = [3, 3, 100, 100];

async function main() {
  await tf.ready();
  await tf.setBackend('webgpu');
  document.getElementById('backend').innerText = `TF.js sets backend as WebGPU`;
  document.getElementById('size').innerText = `conv input dims: [${inputDims}] and filter dims: [${filterDims}]`;
  await tfConv2d(inputDims, filterDims);
  await WebNNConvCPUWithTf(inputDims, filterDims);
  await WebNNConvGPUWithTf(inputDims, filterDims);
  await WebNNConvGPU(inputDims, filterDims);
}

document.getElementById('start').addEventListener('click', () => {main();})