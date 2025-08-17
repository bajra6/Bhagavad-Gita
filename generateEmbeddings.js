// generate-embeddings.js

const fs = require('fs').promises;
const { GoogleGenerativeAI } = require('@google/generative-ai');
const pdf = require('pdf-parse');
require('dotenv').config(); // Use dotenv to load API key locally

// --- Configuration ---
const API_KEY = process.env.GEMINI_API_KEY; // Make sure to set this in a .env file
const PDF_PATH = './Bhagavad Gita 2.pdf';
const OUTPUT_PATH = './gita-embeddings.json'; // The output file

// --- Initialization ---
const genAI = new GoogleGenerativeAI(API_KEY);
const embeddingModel = genAI.getGenerativeModel({ model: 'text-embedding-004' });

// --- Helper Functions (Copied from server.js) ---
const chunkText = (text, chunkSize = 1500) => {
    const chunks = [];
    for (let i = 0; i < text.length; i += chunkSize) {
        chunks.push(text.substring(i, i + chunkSize));
    }
    return chunks;
};

const sanitizeText = (text) => {
    return text
    .replace(/[^\p{L}\p{N}\p{P}\p{Zs}\n\r\t]/gu, ' ')
    .replace(/[ \t]+/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
};

// --- Main Processing Function ---
const createEmbeddingsFile = async () => {
    try {
        console.log('Starting PDF processing...');
        const dataBuffer = await fs.readFile(PDF_PATH);
        const data = await pdf(dataBuffer);
        const chunks = chunkText(data.text);
        
        const nonEmptyChunks = chunks.map(sanitizeText).filter(chunk => chunk.length > 0);

        console.log(`PDF processed. ${nonEmptyChunks.length} valid chunks found. Creating embeddings...`);

        if (nonEmptyChunks.length === 0) throw new Error("PDF processing resulted in no valid text chunks.");

        const BATCH_SIZE = 99;
        let allEmbeddings = [];

        for (let i = 0; i < nonEmptyChunks.length; i += BATCH_SIZE) {
            const batch = nonEmptyChunks.slice(i, i + BATCH_SIZE);
            console.log(`Processing batch ${Math.floor(i / BATCH_SIZE) + 1}...`);
            
            const embeddingResult = await embeddingModel.batchEmbedContents({
                requests: batch.map(chunk => ({ content: { parts: [{ text: chunk }] }, taskType: "RETRIEVAL_DOCUMENT" }))
            });

            allEmbeddings.push(...embeddingResult.embeddings);
        }

        const pdfChunksWithEmbeddings = nonEmptyChunks.map((chunk, index) => ({
            text: chunk,
            embedding: allEmbeddings[index].values,
        }));

        await fs.writeFile(OUTPUT_PATH, JSON.stringify(pdfChunksWithEmbeddings, null, 2));
        console.log(`✅ Success! Embeddings saved to ${OUTPUT_PATH}`);

    } catch (error) {
        console.error(`❌ FATAL: Error creating embeddings file: ${error.message}`, error);
        process.exit(1);
    }
};

createEmbeddingsFile();