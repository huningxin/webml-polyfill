export default class Operation {
  constructor(type, inputs) {
    this._type = type;
    this._inputs = inputs;
    this._outputs = [];
  }

  _addOutput(output) {
    this._outputs.push(output);
  }
}