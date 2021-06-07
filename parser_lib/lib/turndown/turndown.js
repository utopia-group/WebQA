#!/usr/bin/env node

/**
 * Command line utility to allow us to use turndown from python.
 */

const TurndownService = require('turndown')
const fs = require('fs')

const turndown = new TurndownService()

if (process.argv.length < 4) {
    console.log('Usage: ./turndown.js <input file> <output file>')
    process.exit(1)
}

const input = process.argv[2]
const output = process.argv[3]

const contents = fs.readFileSync(input, 'utf8')

const markdown = turndown.turndown(contents)

fs.writeFileSync(output, markdown)
