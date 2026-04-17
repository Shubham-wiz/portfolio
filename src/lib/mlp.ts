// Tiny multilayer perceptron written from scratch.
// Used by the in-browser "Neural Net Playground" demo.

export type Activation = 'tanh' | 'relu' | 'sigmoid';

function act(x: number, a: Activation): number {
  if (a === 'tanh') return Math.tanh(x);
  if (a === 'relu') return x > 0 ? x : 0;
  return 1 / (1 + Math.exp(-x));
}

function actDeriv(y: number, a: Activation): number {
  if (a === 'tanh') return 1 - y * y;
  if (a === 'relu') return y > 0 ? 1 : 0;
  return y * (1 - y);
}

export class MLP {
  // weights[l][j][i] is weight from neuron i (layer l-1) to neuron j (layer l), +1 bias
  weights: number[][][] = [];
  outputs: number[][] = [];
  grads: number[][] = [];
  layers: number[];
  activation: Activation;

  constructor(layers: number[], activation: Activation = 'tanh') {
    this.layers = layers;
    this.activation = activation;
    for (let l = 1; l < layers.length; l++) {
      const W: number[][] = [];
      for (let j = 0; j < layers[l]; j++) {
        const row: number[] = [];
        // Xavier-ish init
        const scale = Math.sqrt(2 / (layers[l - 1] + layers[l]));
        for (let i = 0; i <= layers[l - 1]; i++) {
          row.push((Math.random() * 2 - 1) * scale);
        }
        W.push(row);
      }
      this.weights.push(W);
    }
  }

  forward(input: number[]): number[] {
    this.outputs = [input.slice()];
    for (let l = 0; l < this.weights.length; l++) {
      const prev = this.outputs[l];
      const out: number[] = [];
      const isLast = l === this.weights.length - 1;
      for (let j = 0; j < this.weights[l].length; j++) {
        let z = this.weights[l][j][0]; // bias
        for (let i = 0; i < prev.length; i++) {
          z += prev[i] * this.weights[l][j][i + 1];
        }
        // last layer uses sigmoid for binary classification
        out.push(isLast ? 1 / (1 + Math.exp(-z)) : act(z, this.activation));
      }
      this.outputs.push(out);
    }
    return this.outputs[this.outputs.length - 1];
  }

  // target: 0 or 1 single-output
  trainSample(input: number[], target: number, lr: number): number {
    const out = this.forward(input);
    const loss = -(target * Math.log(out[0] + 1e-9) + (1 - target) * Math.log(1 - out[0] + 1e-9));

    // delta for output layer (sigmoid + BCE => out - target)
    const deltas: number[][] = [];
    for (let l = 0; l < this.weights.length; l++) deltas.push([]);
    const lastL = this.weights.length - 1;
    deltas[lastL] = [out[0] - target];

    for (let l = lastL - 1; l >= 0; l--) {
      const nextW = this.weights[l + 1];
      const nextDelta = deltas[l + 1];
      const outL = this.outputs[l + 1];
      const layerSize = this.weights[l].length;
      for (let i = 0; i < layerSize; i++) {
        let s = 0;
        for (let j = 0; j < nextW.length; j++) {
          s += nextW[j][i + 1] * nextDelta[j];
        }
        deltas[l].push(s * actDeriv(outL[i], this.activation));
      }
    }

    // update
    for (let l = 0; l < this.weights.length; l++) {
      const prev = this.outputs[l];
      for (let j = 0; j < this.weights[l].length; j++) {
        this.weights[l][j][0] -= lr * deltas[l][j];
        for (let i = 0; i < prev.length; i++) {
          this.weights[l][j][i + 1] -= lr * deltas[l][j] * prev[i];
        }
      }
    }
    return loss;
  }
}
