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

const iterations = 10;

async function tfConv2d(inputDims,filterDims){
  tf.setBackend('webgpu');
  await tf.ready();
  const input=tf.ones(inputDims);
  const filter=tf.ones(filterDims);
  const bias = tf.ones([filterDims[3]]);
  let res=tf.conv2d(input,filter,1,'same');
  let result = await res.data();
  let start=performance.now();
  for(let i=0;i<iterations;i++){
    const im = tf.conv2d(input,filter,1,'same');
    res = im.add(bias);
    result = await res.data();
  }
  console.log(result);
  let timeCost=(performance.now()-start)/iterations;
  document.getElementById('op1').innerText ='tfConv2d cost :'+timeCost+'ms';
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

  const input=tf.ones(inputDims);
  const output=tf.zeros(inputDims);

  let inputBuffer=await tf.backend().getGPUBuffer(input.dataId);
  let outputBuffer = await tf.backend().getGPUBuffer(output.dataId);

  let start = performance.now();
  let result;
  for (let i=0;i<iterations;i++) {
    const commandEncoder = device.createCommandEncoder();
    commandEncoder.setNnGraphInput(inputBuffer, 0, execution);
    commandEncoder.setNnGraphOutput(outputBuffer, 0, execution);
    commandEncoder.executeNnGraph(execution);
    device.getQueue().submit([commandEncoder.finish()]);
  
    result = await output.data();
  }
  console.log(result);
  let timeCost=(performance.now()-start)/(iterations-1);

  document.getElementById('op2').innerText ='WebML conv2d cost'+timeCost+'ms';

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


