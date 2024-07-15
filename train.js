#!/usr/bin/env node

import fs from 'node:fs/promises'
import path from 'node:path'
import {existsSync} from 'node:fs'
import prettyMilliseconds from 'pretty-ms'
import {ParquetReader} from '@dsnp/parquetjs'
import {TokenizerLoader} from '@lenml/tokenizers'
import {core as mx, optimizers as optim, nn, utils} from '@frost-beta/mlx'

import Model from './model.js'

if (process.argv.length < 3) {
  console.error('Usage: train.js /path/to/train-*.parquet')
  process.exit(0)
}

// Hyperparameters.
const contextSize = 256

// Traning configs.
const epochs = 1
const batchSize = 32
const learningRate = 1e-4
const maxRows = Infinity

main()

async function main() {
  // Read the config.json from current dir and create the model from it.
  const config = JSON.parse(await fs.readFile('config.json'))
  const model = new Model(config)

  // Continue last training.
  const weightsFile = 'weights.safetensors'
  if (existsSync(weightsFile)) {
    console.log(`Loading weights from ${weightsFile}.`)
    model.loadWeights(weightsFile)
  }

  // Load our own tokenizer.
  const tokenizer = TokenizerLoader.fromPreTrained({
    tokenizerJSON: JSON.parse(await fs.readFile('tokenizer.json')),
    tokenizerConfig: JSON.parse(await fs.readFile('tokenizer_config.json')),
  })

  // Calculate how many parameters the model has.
  let nparams = 0
  for (const [k, x] of utils.treeFlatten(model.parameters())) {
    if (!k.includes('embedTokens'))
      nparams += x.size
  }
  console.log(`Training Llama3 with ${(nparams / 1024 ** 2).toFixed(1)}M parameters.`)

  // Command line flags.
  const files = shuffle(process.argv.slice(2))
  const totalRows = Math.min(maxRows, await getRowCount(files))
  const reportPerIter = Math.max(Math.floor(32 / batchSize * 10), 1)
  console.log('Total rows of data to train:', totalRows)

  // Preprare utils for doing gradient descent.
  const lossAndGradFunction = nn.valueAndGrad(model, lossFunction)
  const optimizer = new optim.AdamW(learningRate)

  // Read batches from the datasets.
  let lastRow = 0
  let losses = []
  for (let e = 0, iterations = 0, start = Date.now(); e < epochs; ++e) {
    for await (const [row, x, y] of iterateBatches(files, tokenizer, contextSize, batchSize)) {
      if (lastRow > maxRows)
        break
      // Use mx.tidy to free all the intermediate tensors immediately.
      mx.tidy(() => {
        // Compute loss and gradients, then update the model.
        const [loss, grads] = lossAndGradFunction(model, mx.array(x, mx.int32), mx.array(y, mx.int32))
        optimizer.update(model, grads)
        mx.eval(model.state, optimizer.state)
        losses.push(loss.item())
        // Keep the states of model and optimizer from getting freed.
        return [model.state, optimizer.state]
      })
      // Report updates.
      if (++iterations % reportPerIter === 0) {
        const stop = Date.now()
        const trainLoss = mean(losses)
        const processedRows = e * totalRows + row
        const eta = ((epochs * totalRows - processedRows) / (processedRows - lastRow)) * (stop - start)
        console.log(`Iter ${iterations}`,
                    `(${(100 * processedRows / (epochs * totalRows)).toFixed(1)}%):`,
                    `Train loss ${trainLoss.toFixed(2)},`,
                    `It/sec ${(reportPerIter / (stop - start) * 1000).toFixed(2)},`,
                    `ETA ${prettyMilliseconds(eta, {compact: true})}.`)
        start = Date.now()
        losses = []
        lastRow = processedRows
      }
    }
  }

  // Save weights on exit.
  console.log(`Save weights to ${weightsFile}.`)
  model.saveWeights(weightsFile)
}

// Return the total number of rows.
async function getRowCount(files) {
  let count = 0
  for (const f of files) {
    const reader = await ParquetReader.openFile(f)
    count += parseInt(reader.getRowCount())
    await reader.close()
  }
  return count
}

// Read datasets from |files|, and generate batches of [inputs, targets].
async function* iterateBatches(files, tokenizer, contextSize, batchSize) {
  let row = 0
  let xBatches = []
  let yBatches = []
  for (const f of files) {
    // Read the dataset.
    const reader = await ParquetReader.openFile(f)
    const cursor = reader.getCursor()
    let record
    while (record = await cursor.next()) {
      ++row
      // Convert text to tokens.
      const tokens = tokenizer.encode(record.text)
      // Generate batches from the tokens.
      for (let i = 0; i < tokens.length - 1; i += contextSize) {
        const length = Math.min(contextSize, tokens.length - i - 1)
        // If the batch's length is less than contextSize, fill it with EOS.
        let paddings = []
        if (length < contextSize)
          paddings = new Array(contextSize - length).fill(tokenizer.model.eosTokenId)
        xBatches.push(tokens.slice(i, i + length).concat(paddings))
        yBatches.push(tokens.slice(i + 1, i + 1 + length).concat(paddings))
      }
      // Yield batches with each batch of |batchSize|.
      while (xBatches.length >= batchSize) {
        yield [ row, xBatches.splice(0, batchSize), yBatches.splice(0, batchSize) ]
      }
    }
    await reader.close()
  }
}

// Calculate the loss by 1) running the model with the inputs, and 2) then using
// cross entropy function to get the loss between the results and targets.
function lossFunction(model, x, y) {
  const [logits, cache] = model.forward(x)
  const losses = nn.losses.crossEntropy(logits, y)
  return mx.mean(losses)
}

// Compute the mean value of a JS array.
function mean(array) {
  if (array.length == 0)
    return 0
  return array.reduce((a, b) => a + b) / array.length
}

// Shuffle the elements in a JS array.
function shuffle(array) {
  const result = array.slice()
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]]
  }
  return result
}
