import {OperationCode, OperandCode, PaddingCode, PreferenceCode, FuseCode, ResultCode} from './Enums'
import Model from './Model'
import Compilation from './Compilation'
import Execution from './Execution'
import WebGLModel from './webgl/WebGLModel'
import Operand from './Operand'
import Operation from './Operation'

export default class NeuralNetworkContext {
  constructor() {
    this._initOperandTypes();
    this._initOperationTypes();
    this._initFusedActivationFunctionTypes();
    this._initImplicitPaddingTypes();
    this._initExecutionPreferenceTypes();
    this.supportWebGL = WebGLModel._supportWebGL();
    this.supportWasm = !!window.WebAssembly;
  }

  /**
   * Create a model object.
   * 
   * @param {options} options.backend - model backend.
   */
  async createModel(options = {}) {
    if (options.backend === 'WebGL' && !this.supportWebGL) {
      return "WebGL is not available";
    } else if (!this.supportWasm) {
      return "WebAssembly is not available";
    }
    return new Model(options);
  }

  async createModelByOutputs(outputs, options = {}) {
    let operands = [];
    let operations = [];
    function handleOperation(operation) {
      operations.push(operation);
      for (let i in operation._inputs) {
        handleOperand(operation._inputs[i]);
      }
    }
    function handleOperand(operand) {
      operands.push(operand);
      if (operand._operation) {
        handleOperation(operand._operation);
      }
    }
    for (let i in outputs) {
      handleOperand(outputs[i]);
    }
    let model = new Model(options);
    let inputs = [];
    for (let i in operands) {
      const operand = operands[i];
      model.addOperand(operand._desc);
      if (!operand._value && !operand._operation) {
        inputs.push(operand);
      } else if (operand._value) {
        model.setOperandValue(i, operand._value);
      }
      operand._setIndex(i);
    }
    for (let i in operations) {
      const operation = operations[i];
      const ins = operation._inputs.map((operand) => {return operand._index;});
      const outs = operation._outputs.map((operand) => {return operand._index;});
      model.addOperation(operation._type, ins, outs);
    }
    const inputIndex = inputs.map((operand) => {return operand._index;});
    const outputIndex = outputs.map((operand) => {return operand._index;});
    model.identifyInputsAndOutputs(inputIndex, outputIndex);
    await model.finish();
    return model;
  }


  input(desc) {
    return new Operand(desc);
  }

  constant(desc, value) {
    let constant = new Operand(desc);
    constant._setValue(value);
    return constant;
  }

  add(a, b) {
    const fuseCode = this.constant({type: this.INT32}, new Int32Array([nn.FUSED_NONE]));
    let result = new Operand(a._desc);
    result._setOperation(new Operation(OperationCode.ADD, [a, b, fuseCode]));
    return result;
  }

  mul(a, b) {
    const fuseCode = this.constant({type: this.INT32}, new Int32Array([nn.FUSED_NONE]));
    let result = new Operand(a._desc);
    result._setOperation(new Operation(OperationCode.MUL, [a, b, fuseCode]));
    return result;
  }

  _initOperandTypes() {
    this.FLOAT32 = OperandCode.FLOAT32;
    this.INT32 = OperandCode.INT32;
    this.UINT32 = OperandCode.UINT32;
    this.TENSOR_FLOAT32 = OperandCode.TENSOR_FLOAT32;
    this.TENSOR_INT32 = OperandCode.TENSOR_INT32;
    this.TENSOR_QUANT8_ASYMM = OperandCode.TENSOR_QUANT8_ASYMM;
  }

  _initOperationTypes() {
    this.ADD = OperationCode.ADD;
    this.AVERAGE_POOL_2D = OperationCode.AVERAGE_POOL_2D;
    this.CONCATENATION = OperationCode.CONCATENATION;
    this.CONV_2D = OperationCode.CONV_2D;
    this.DEPTHWISE_CONV_2D = OperationCode.DEPTHWISE_CONV_2D;
    this.DEPTH_TO_SPACE = OperationCode.DEPTH_TO_SPACE;
    this.DEQUANTIZE = OperationCode.DEQUANTIZE;
    this.EMBEDDING_LOOKUP = OperationCode.EMBEDDING_LOOKUP;
    this.FLOOR = OperationCode.FLOOR;
    this.FULLY_CONNECTED = OperationCode.FULLY_CONNECTED;
    this.HASHTABLE_LOOKUP = OperationCode.HASHTABLE_LOOKUP;
    this.L2_NORMALIZATION = OperationCode.L2_NORMALIZATION;
    this.L2_POOL_2D = OperationCode.L2_POOL_2D;
    this.LOCAL_RESPONSE_NORMALIZATION = OperationCode.LOCAL_RESPONSE_NORMALIZATION;
    this.LOGISTIC = OperationCode.LOGISTIC;
    this.LSH_PROJECTION = OperationCode.LSH_PROJECTION;
    this.LSTM = OperationCode.LSTM;
    this.MAX_POOL_2D = OperationCode.MAX_POOL_2D;
    this.MUL = OperationCode.MUL;
    this.RELU = OperationCode.RELU;
    this.RELU1 = OperationCode.RELU1;
    this.RELU6 = OperationCode.RELU6;
    this.RESHAPE = OperationCode.RESHAPE;
    this.RESIZE_BILINEAR = OperationCode.RESIZE_BILINEAR;
    this.RNN = OperationCode.RNN;
    this.SOFTMAX = OperationCode.SOFTMAX;
    this.SPACE_TO_DEPTH = OperationCode.SPACE_TO_DEPTH;
    this.SVDF = OperationCode.SVDF;
    this.TANH = OperationCode.TANH;
    this.BATCH_TO_SPACE_ND = OperationCode.BATCH_TO_SPACE_ND;
    this.TRANSPOSE = OperationCode.TRANSPOSE;
    this.MAXIMUM = OperationCode.MAXIMUM;
    this.ATROUS_CONV_2D = OperationCode.ATROUS_CONV_2D;
    this.ATROUS_DEPTHWISE_CONV_2D = OperationCode.ATROUS_DEPTHWISE_CONV_2D;
  }

  _initFusedActivationFunctionTypes() {
    this.FUSED_NONE = FuseCode.NONE;
    this.FUSED_RELU = FuseCode.RELU;
    this.FUSED_RELU1 = FuseCode.RELU1;
    this.FUSED_RELU6 = FuseCode.RELU6;
  }

  _initImplicitPaddingTypes() {
    this.PADDING_SAME = PaddingCode.SAME;
    this.PADDING_VALID = PaddingCode.VALID;
  }

  _initExecutionPreferenceTypes() {
    this.PREFER_LOW_POWER = PreferenceCode.LOW_POWER;
    this.PREFER_FAST_SINGLE_ANSWER = PreferenceCode.FAST_SINGLE_ANSWER;
    this.PREFER_SUSTAINED_SPEED = PreferenceCode.SUSTAINED_SPEED;
  }

  _initResultCodes() {
    this.NO_ERROR = ResultCode.NO_ERROR;
    this.OUT_OF_MEMORY = ResultCode.OUT_OF_MEMORY;
    this.INCOMPLETE = ResultCode.INCOMPLETE;
    this.UNEXPECTED_NULL = ResultCode.UNEXPECTED_NULL;
    this.BAD_DATA = ResultCode.BAD_DATA;
    this.OP_FAILED = ResultCode.OP_FAILED;
    this.UNMAPPABLE = ResultCode.UNMAPPABLE;
    this.BAD_STATE = ResultCode.BAD_STATE;
  }
}
