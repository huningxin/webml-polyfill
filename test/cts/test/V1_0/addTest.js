function product(shape) {
  let result = 1;
  for (let i = 0; i < shape.length; i++) {
    result = result * shape[i];
  }
  return result;
}

async function executeWithWebGPU() {
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice({});
  const commandEncoder = device.createCommandEncoder({});

  // Get a GPU buffer and an arrayBuffer for writing.
  // Upon success the GPU buffer is returned in the mapped state.
  const [gpuWriteBuffer, arrayBuffer] = device.createBufferMapped({
    size: 8,
    usage: GPUBufferUsage.MAP_WRITE
  });

  // Write bytes to buffer.
  new Float32Array(arrayBuffer).set([3, 4]);
  // Unmap buffer so that it can be used later for copy.
  gpuWriteBuffer.unmap();

  // Unmap buffer so that it can be used later for copy.
  commandEncoder.shareBufferToWebml(gpuWriteBuffer);

  device.getQueue().submit([commandEncoder.finish()]);
}
async function bufferTest(){
  tf.setBackend('webgpu');
  await tf.ready();
  let device = tf.backend().device;
  let a=tf.tensor1d([1,2,3,4], 'float32');
  let buffer=await tf.backend().getGPUBuffer(a.dataId);
  gpuReadBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
  
    const commandEncoder = device.createCommandEncoder({});
    commandEncoder.copyBufferToBuffer(
      buffer /* source buffer */,
      0 /* source offset */,
      gpuReadBuffer /* destination buffer */,
      0 /* destination offset */,
      16 /* size */
    );
    device.getQueue().submit([commandEncoder.finish()]);
    const copyArrayBuffer = await gpuReadBuffer.mapReadAsync();
    console.log(new Float32Array(copyArrayBuffer));
}
async function tfAdd(inputDims) {
  tf.setBackend('webgpu');
  await tf.ready();

  const input1=tf.truncatedNormal(inputDims,1);
  const input2=tf.truncatedNormal(inputDims,1);
  let res=tf.add(input1,input2);
  await tf.data();
  let start=performance.now();
  for (let i=0;i<5;++i){  
    res=tf.add(input1,input2);
    await res.data();
  }
  let timeCost=(performance.now()-start)/5;
  document.getElementById('op1').innerText='Op add cost'+timeCost+'ms';
}

const iterations = 100;

async function tfConv2d(inputDims,filterDims){
  tf.setBackend('webgpu');
  await tf.ready();
  const input = tf.ones(inputDims);
  const filter = tf.ones(filterDims);
  const bias = tf.ones([filterDims[3]]);
  //warm up
  let im0 = tf.conv2d(input, filter, 1, 'same');
  let im1 = tf.add(im0, bias);
  let im2 = tf.relu(im1);
  let result = await im2.data();
  let start = performance.now();
  for(let i=0; i<iterations; i++){
    m0 = tf.conv2d(input, filter, 1, 'same');
    im1 = tf.add(im0, bias);
    im2 = tf.relu(im1);
    result = await im2.data();
  }
  console.log(result);
  let elapsedTime = ((performance.now() - start) / iterations).toFixed(2);
  document.getElementById('op1').innerText =`tfConv2d elapsed time: ${elapsedTime} ms`;
}

async function tfConv2dx2(inputDims,filterDims){
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
  document.getElementById('op1').innerText =`tfConv2d elapsed time: ${elapsedTime} ms`;
}

async function WebNNConvGPU(inputDims,filterDims) {
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
  const output = operandIndex++;
  model.addOperand(inputDesc);

  const filterData = await tf.ones(filterDims).data();
  const biasData = await tf.ones([filterDims[3]]).data();
  model.setOperandValue(filter, filterData);
  model.setOperandValue(bias, biasData);
  model.setOperandValue(pad, new Int32Array([(filterDims[1]-1)/2]));
  model.setOperandValue(act, new Int32Array([nn.FUSE_RELU]));
  model.setOperandValue(stride, new Int32Array([1]));
  model.addOperation(nn.CONV_2D, [input, filter, bias, pad, pad, pad, pad, stride, stride, act], [output]);

  model.identifyInputsAndOutputs([input], [output]);
  await model.finish();
  let compilation = await model.createCompilation();
  compilation.setPreference(nn.PREFER_SUSTAINED_SPEED);
  await compilation.finish();
  let execution = await compilation.createExecution();

  const inputTensor = tf.ones(inputDims);
  const outputTensor = tf.zeros(inputDims);

  let inputBuffer = await tf.backend().getGPUBuffer(inputTensor.dataId);
  let outputBuffer = await tf.backend().getGPUBuffer(outputTensor.dataId);

  let start = performance.now();
  let result;
  for (let i=0;i<iterations;i++) {
    const commandEncoder = device.createCommandEncoder();
    commandEncoder.setNnGraphInput(inputBuffer, 0, execution);
    commandEncoder.setNnGraphOutput(outputBuffer, 0, execution);
    commandEncoder.executeNnGraph(execution);
    device.getQueue().submit([commandEncoder.finish()]);
    result = await outputTensor.data();
  }
  console.log(result);

  const  elapsedTime =((performance.now() - start) / iterations).toFixed(2);
  document.getElementById('op2').innerText = `WebNN conv elapsed time: ${elapsedTime} ms`;
}

async function WebNNConvGPUWithTf(inputDims,filterDims) {
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
  const output = operandIndex++;
  model.addOperand(inputDesc);

  const filterData = await tf.ones(filterDims).data();
  const biasData = await tf.zeros([filterDims[3]]).data();
  model.setOperandValue(filter, filterData);
  model.setOperandValue(bias, biasData);
  model.setOperandValue(pad, new Int32Array([(filterDims[1]-1)/2]));
  model.setOperandValue(act, new Int32Array([nn.FUSE_NONE]));
  model.setOperandValue(stride, new Int32Array([1]));
  model.addOperation(nn.CONV_2D, [input, filter, bias, pad, pad, pad, pad, stride, stride, act], [output]);

  model.identifyInputsAndOutputs([input], [output]);
  await model.finish();
  let compilation = await model.createCompilation();
  compilation.setPreference(nn.PREFER_SUSTAINED_SPEED);
  await compilation.finish();
  let execution = await compilation.createExecution();

  const inputTensor = tf.ones(inputDims);
  const outputTensor = tf.zeros(inputDims);

  let inputBuffer = await tf.backend().getGPUBuffer(inputTensor.dataId);
  let outputBuffer = await tf.backend().getGPUBuffer(outputTensor.dataId);

  let commandEncoder = device.createCommandEncoder();
  commandEncoder.setNnGraphInput(inputBuffer, 0, execution);
  commandEncoder.setNnGraphOutput(outputBuffer, 0, execution);
  commandEncoder.executeNnGraph(execution);
  device.getQueue().submit([commandEncoder.finish()]);
  const biasTensor = tf.ones([filterDims[3]]);
  let im0 = tf.add(outputTensor, biasTensor);
  let im1 = tf.relu(im0);
  let result = await im1.data();
  let start = performance.now();
  for (let i=0;i<iterations;i++) {
    commandEncoder = device.createCommandEncoder();
    commandEncoder.setNnGraphInput(inputBuffer, 0, execution);
    commandEncoder.setNnGraphOutput(outputBuffer, 0, execution);
    commandEncoder.executeNnGraph(execution);
    device.getQueue().submit([commandEncoder.finish()]);
    im0 = tf.add(outputTensor, biasTensor);
    im1 = tf.relu(im0);
    result = await im1.data();
  }
  console.log(result);

  const  elapsedTime =((performance.now() - start) / iterations).toFixed(2);
  document.getElementById('op2').innerText = `WebNN conv elapsed time: ${elapsedTime} ms`;
}

async function WebNNConvGPUx2(inputDims,filterDims) {
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
    const commandEncoder = device.createCommandEncoder();
    commandEncoder.setNnGraphInput(inputBuffer, 0, execution);
    commandEncoder.setNnGraphOutput(outputBuffer, 0, execution);
    commandEncoder.executeNnGraph(execution);
    device.getQueue().submit([commandEncoder.finish()]);
    result = await outputTensor.data();
  }
  console.log(result);

  const  elapsedTime =((performance.now() - start) / iterations).toFixed(2);
  document.getElementById('op2').innerText = `WebNN conv elapsed time: ${elapsedTime} ms`;
}

async function createModel() {
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
  const biasData = await tf.ones([filterDims[3]]).data();
  model.setOperandValue(filter, filterData);
  model.setOperandValue(bias, biasData);
  model.setOperandValue(pad, new Int32Array([(filterDims[1]-1)/2]));
  model.setOperandValue(act, new Int32Array([nn.FUSE_RELU]));
  model.setOperandValue(stride, new Int32Array([1]));
  model.addOperation(nn.CONV_2D, [input, filter, bias, pad, pad, pad, pad, stride, stride, act], [output]);

  model.identifyInputsAndOutputs([input], [output]);
  await model.finish();
  return model;
}

async function WebNNConvGPUx2Model(inputDims,filterDims) {
  tf.setBackend('webgpu');
  await tf.ready();
  let device=tf.backend().device;
  const nn = navigator.ml.getNeuralNetworkContext();

  let model1 = await createModel();
  let compilation1 = await model1.createCompilation();
  compilation1.setPreference(nn.PREFER_SUSTAINED_SPEED);
  await compilation1.finish();
  let execution1 = await compilation1.createExecution();

  let model2 = await createModel();
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
  document.getElementById('op2').innerText = `WebNN conv elapsed time: ${elapsedTime} ms`;
}

async function WebNNConvGPUx2WithTf(inputDims,filterDims) {
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
  const output = operandIndex++;
  model.addOperand(inputDesc);

  const filterData = await tf.ones(filterDims).data();
  const biasData = await tf.ones([filterDims[3]]).data();
  model.setOperandValue(filter, filterData);
  model.setOperandValue(bias, biasData);
  model.setOperandValue(pad, new Int32Array([(filterDims[1]-1)/2]));
  model.setOperandValue(act, new Int32Array([nn.FUSE_RELU]));
  model.setOperandValue(stride, new Int32Array([1]));
  model.addOperation(nn.CONV_2D, [input, filter, bias, pad, pad, pad, pad, stride, stride, act], [output]);

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

  const filterTensor = tf.ones(filterDims);
  const biasTensor = tf.ones([filterDims[3]]);
  // let im0 = tf.conv2d(inputTensor, filterTensor, 1, 'same');
  // let im1 = tf.add(im0, biasTensor);
  // let im2 = tf.relu(im1);
  let inputBuffer = await tf.backend().getGPUBuffer(inputTensor.dataId);
  let outputBuffer = await tf.backend().getGPUBuffer(outputTensor.dataId);
  let commandEncoder = device.createCommandEncoder();
  commandEncoder.setNnGraphInput(inputBuffer, 0, execution);
  commandEncoder.setNnGraphOutput(outputBuffer, 0, execution);
  commandEncoder.executeNnGraph(execution);
  device.getQueue().submit([commandEncoder.finish()]);
  let im0 = tf.conv2d(outputTensor, filterTensor, 1, 'same');
  let im1 = tf.add(im0, biasTensor);
  let im2 = tf.relu(im1);
  let result = await im2.data();
  let start = performance.now();
  for (let i=0;i<iterations;i++) {
    commandEncoder = device.createCommandEncoder();
    commandEncoder.setNnGraphInput(inputBuffer, 0, execution);
    commandEncoder.setNnGraphOutput(outputBuffer, 0, execution);
    commandEncoder.executeNnGraph(execution);
    device.getQueue().submit([commandEncoder.finish()]);
    im0 = tf.conv2d(outputTensor, filterTensor, 1, 'same');
    im1 = tf.add(im0, biasTensor);
    im2 = tf.relu(im1);
    result = await im2.data();
  }
  console.log(result);

  const  elapsedTime =((performance.now() - start) / iterations).toFixed(2);
  document.getElementById('op2').innerText = `WebNN conv elapsed time: ${elapsedTime} ms`;
}

async function WebNNConvCPU(inputDims,filterDims) {
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
  let type0 = {type: nn.TENSOR_FLOAT32, dimensions: inputDims};
  let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [filterDims[3], filterDims[0], filterDims[1], filterDims[2]]};
  let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [filterDims[3]]};
  let type3 = {type: nn.INT32};

  let op1 = operandIndex++;
  model.addOperand(type0);
  let op2 = operandIndex++;
  model.addOperand(type1);
  let op3 = operandIndex++;
  model.addOperand(type2);
  let pad0 = operandIndex++;
  model.addOperand(type3);
  let act = operandIndex++;
  model.addOperand(type3);
  let stride = operandIndex++;
  model.addOperand(type3);
  let op4 = operandIndex++;
  model.addOperand(type0);

  const filter=await tf.ones(filterDims).data();
  const bias = await tf.ones([filterDims[3]]).data();
  model.setOperandValue(op2, filter);
  model.setOperandValue(op3, bias);
  model.setOperandValue(pad0, new Int32Array([(filterDims[1]-1)/2]));
  model.setOperandValue(act, new Int32Array([0]));
  model.setOperandValue(stride, new Int32Array([1]));
  model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);

  model.identifyInputsAndOutputs([op1], [op4]);
  await model.finish();

  let compilation = await model.createCompilation();
  compilation.setPreference(nn.PREFER_SUSTAINED_SPEED);
  await compilation.finish();
  let execution = await compilation.createExecution();

  const input=await tf.ones(inputDims).data();
  const output=await tf.zeros(inputDims).data();

  execution.setInput(0, input);
  execution.setOutput(0, output);

  let start = performance.now();
  for (let i=0;i<iterations;i++) {
    result = await execution.startCompute();
  }
  console.log(output);
  let timeCost=(performance.now()-start)/(iterations-1);

  document.getElementById('op2').innerText ='WebML conv2d cost'+timeCost+'ms';

}

async function tfAddAndConv2d(inputDims,filterDims) {
  tf.setBackend('webgpu');
  await tf.ready();
  const input1=tf.truncatedNormal(inputDims,1);
  const input2=tf.truncatedNormal(inputDims,1);
  const filter=tf.truncatedNormal(filterDims,1);
  let res1=tf.add(input1,input2);
  await res1.data()
  let res2= tf.conv2d(res1,filter,1,'valid');
  await res2.data();
  let start=performance.now();
  for(let i=0;i<5;++i) {
    res1= tf.add(input1,input2);
    await res1.data()
    res2= tf.conv2d(res1,filter,1,'valid');
    await res2.data();
  }
  let timeCost=(performance.now()-start)/5;
  document.getElementById('op1').innerText ='Ops conv2d cost :'+timeCost+'ms';

}

async function tfModel(inputDims,filterDims){
  tf.setBackend('webgl');  //webgpu backend is not supported
  await tf.ready();
  const model = tf.sequential();
  model.add(tf.layers.conv2d({
    batchInputShape: inputDims,
    kernelSize: [filterDims[1],filterDims[2]],
    filters: filterDims[0],
    strides : 1,
    padding : 'valid',
    dataFormat : 'channelsLast',
    activation: 'relu',
    dtype : 'float32'
  }));
  model.add(tf.layers.conv2d({
    kernelSize: [filterDims[1],filterDims[2]], 
    filters: filterDims[0], 
    strides : 1,
    padding : 'valid',
    dataFormat : 'channelsLast',
    activation: 'relu',
    dtype : 'float32'
  }));
  const input=tf.truncatedNormal(inputDims,1);
  let start=performance.now();
  for(let i=0;i<5;i++){
  tf.tidy(()=>model.predict(input)  )}
  let timeCost=(performance.now()-start)/5;
  document.getElementById('op1').innerText ='model cost :'+timeCost+'ms';
}

async function dualTest(inputDims,filterDims) {
  tf.setBackend('webgpu');
  await tf.ready();
  let device=tf.backend().device;
  const nn = navigator.ml.getNeuralNetworkContext();
  const filter=await tf.truncatedNormal(filterDims,1).data();
  const input1=tf.truncatedNormal(inputDims,1);
  const input2=tf.truncatedNormal(inputDims,1);
  let tmp = tf.add(input1,input2);
  await tmp.data();  //warm up

  options={
    "backend": "WebML",
    "prefer": "sustained"
  };
  let model = await nn.createModel(options);
  let operandIndex = 0;

  // inputDims [n,h,w,i]
  // filterDims [h,w,i,o]
  let type0 = {type: nn.TENSOR_FLOAT32, dimensions: inputDims};
  let type0_length = product(type0.dimensions);
  let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [filterDims[3], filterDims[0], filterDims[1], filterDims[2]]};
  let type1_length = product(type1.dimensions);
  let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [filterDims[3]]};
  let type2_length = product(type2.dimensions);
  let type3 = {type: nn.INT32};

  let op1 = operandIndex++;
  model.addOperand(type0);
  let op2 = operandIndex++;
  model.addOperand(type1);
  let op3 = operandIndex++;
  model.addOperand(type2);
  let pad0 = operandIndex++;
  model.addOperand(type3);
  let act = operandIndex++;
  model.addOperand(type3);
  let stride = operandIndex++;
  model.addOperand(type3);
  let op4 = operandIndex++;
  model.addOperand(type0);

  model.setOperandValue(op2, filter);
  model.setOperandValue(op3, new Float32Array([0]));
  model.setOperandValue(pad0, new Int32Array([(filterDims[1]-1)/2]));
  model.setOperandValue(act, new Int32Array([0]));
  model.setOperandValue(stride, new Int32Array([1]));
  model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);

  model.identifyInputsAndOutputs([op1], [op4]);
  await model.finish();

  let compilation = await model.createCompilation();
  compilation.setPreference(getPreferenceCode(options.prefer));
  await compilation.finish();
  let execution = await compilation.createExecution();

  //let op1_input = new Float32Array(op1_value);
  //execution.setInput(0, op1_value);
  let op4_output = new Float32Array(type0_length);
  execution.setOutput(0, op4_output);

  let start=performance.now();
  let res1=tf.add(input1,input2)
  for(let i=0;i<5;i++){
  res1=tf.add(input1,input2);
  await res1.data()
  let gpuWriteBuffer=await tf.backend().getGPUBuffer(res1.dataId);
  let bufferSize=product(inputDims)*4;
  let gpuReadBuffer = device.createBuffer({
    size: bufferSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const commandEncoder = device.createCommandEncoder({});
  commandEncoder.copyBufferToBuffer(
    gpuWriteBuffer /* source buffer */,
    0 /* source offset */,
    gpuReadBuffer /* destination buffer */,
    0 /* destination offset */,
    bufferSize /* size */
  );
  commandEncoder.shareBufferToWebml(gpuReadBuffer, 0);
  device.getQueue().submit([commandEncoder.finish()]);
  await execution.startCompute();}
  let timeCost=((performance.now()-start))/5;

  document.getElementById('op2').innerText ='dualTest cost  :'+timeCost+'ms';

}
const inputDims = [1, 100, 100, 100];
const filterDims = [3, 3, 100, 100];

//tfConv2d(inputDims,filterDims);
//WebMLConv(inputDims,filterDims);
//dualTest([1,10,10,1024],[1,1,1024,1024]);
//tfModel([1,10,10,1024],[1024,1,1,1024]);


