import Operation from './Operation'

export default class Operand {
  constructor(desc) {
    this._desc = desc;
    this._value = null;
    this._operation = null;
    this._index = -1;
  }

  _setValue(value) {
    this._value = value;
  }

  _setOperation(operation) {
    operation._addOutput(this);
    this._operation = operation;
  }

  _setIndex(index) {
    this._index = index;
  }
}
