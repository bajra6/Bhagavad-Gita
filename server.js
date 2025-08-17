// server.js

const express = require('express');
const fs = require('fs').promises;
const { GoogleGenerativeAI } = require('@google/generative-ai');
const pdf = require('pdf-parse');
const NodeCache = require('node-cache');
const cors = require('cors');
require('dotenv').config();
const path = require('path');
const cosineSimilarity = require('cosine-similarity'); // NEW: For vector comparison

// --- Configuration ---
const API_KEY = process.env.GEMINI_API_KEY || 'xxxx';
const PDF_PATH = './Bhagavad Gita 2.pdf'; // UPDATED file path
const PORT = process.env.PORT || 3000;
// const LOG_FILE = './server.log'; // Added for logging

// --- Initialization ---
const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));


const genAI = new GoogleGenerativeAI(API_KEY);
// Model for chat generation
const model = genAI.getGenerativeModel({ model: 'gemini-2.0-flash' }); // UPDATED model
// NEW: Model specifically for creating text embeddings
const embeddingModel = genAI.getGenerativeModel({ model: 'text-embedding-004' });

const sessionCache = new NodeCache({ stdTTL: 3600, checkperiod: 600 });

// MODIFIED: This will now store chunks along with their vector embeddings
let pdfChunksWithEmbeddings = [];

// --- NEW: Logging Function ---
const logToFile = async (message) => {
    const timestamp = new Date().toISOString();
    const logMessage = `${timestamp} - ${message}\n`;
    try {
        await fs.appendFile(LOG_FILE, logMessage);
    } catch (error) {
        console.error('Failed to write to log file:', error);
    }
};

// --- Helper Functions ---
const chunkText = (text, chunkSize = 1500) => { // Using a safe chunk size
    const chunks = [];
    for (let i = 0; i < text.length; i += chunkSize) {
        chunks.push(text.substring(i, i + chunkSize));
    }
    return chunks;
};

/**
 * ✅ **FIX**: A more aggressive sanitization function.
 * This removes all non-printable ASCII characters to prevent API errors.
 * @param {string} text The text to clean.
 * @returns {string} The sanitized text.
 */
const sanitizeText = (text) => {
    return text
    .replace(/[^\p{L}\p{N}\p{P}\p{Zs}\n\r\t]/gu, ' ')
    .replace(/[ \t]+/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
};

/**
 * NEW: Retrieves relevant chunks using vector embeddings for semantic search.
 * This finds chunks based on conceptual meaning, not just keywords.
 * @param {string} query The user's current query.
 * @param {object[]} history The chat history.
 * @param {number} topK The number of top relevant chunks to return.
 * @returns {Promise<string[]>} A promise that resolves to the most relevant text chunks.
 */
const retrieveRelevantChunksVector = async (query, history, topK = 5) => {
    if (pdfChunksWithEmbeddings.length === 0) {
        // await logToFile("WARN: PDF chunks not processed yet. Returning empty context.");
        return [];
    }
    const recentHistoryText = history.slice(-4).map(h => (h?.parts?.text ?? '')).filter(Boolean).join('\n');
    const combinedQuery = `${recentHistoryText} \n ${query}`;

    // 1. Create an embedding for the user's query
    const queryEmbeddingResult = await embeddingModel.embedContent(
        { content: { parts: [{ text: combinedQuery }] }, taskType: "RETRIEVAL_QUERY" }
    );
    const queryEmbedding = queryEmbeddingResult.embedding.values;

    // 2. Compare the query's embedding with all pre-calculated chunk embeddings
    const scoredChunks = pdfChunksWithEmbeddings.map(chunk => ({
        text: chunk.text,
        score: cosineSimilarity(queryEmbedding, chunk.embedding)
    }));

    // 3. Sort by the highest similarity score
    scoredChunks.sort((a, b) => b.score - a.score);

    // 4. Return the text of the top K most similar chunks
    return scoredChunks.slice(0, topK).map(item => item.text);
};

// --- PDF Processing ---
/**
 * MODIFIED: Loads, processes the PDF, and creates vector embeddings for each chunk.
 * This is done once at server startup and now processes in batches.
 */
const processPdf = async () => {
    try {
        // await logToFile('INFO: Starting PDF processing...');
        console.log('Starting PDF processing...');
        const dataBuffer = await fs.readFile(PDF_PATH);
        const data = await pdf(dataBuffer);
        const chunks = chunkText(data.text);
        
        // Sanitize each chunk first, then filter out any that become empty after cleaning.
        const sanitizedChunks = chunks.map(sanitizeText);
        const nonEmptyChunks = sanitizedChunks.filter(chunk => chunk.length > 0);

        // await logToFile(`INFO: PDF processed. ${nonEmptyChunks.length} valid chunks found. Now creating embeddings in batches...`);
        console.log(`INFO: PDF processed. ${nonEmptyChunks.length} valid chunks found. Now creating embeddings in batches...`);

        if (nonEmptyChunks.length === 0) throw new Error("PDF processing resulted in no valid text chunks after sanitization.");

        const BATCH_SIZE = 99; // API limit is 100, use 99 for safety
        let allEmbeddings = [];

        for (let i = 0; i < nonEmptyChunks.length; i += BATCH_SIZE) {
            const batch = nonEmptyChunks.slice(i, i + BATCH_SIZE);
            const batchNumber = Math.floor(i / BATCH_SIZE) + 1;
            // await logToFile(`INFO: Processing batch ${batchNumber}...`);
            console.log(`INFO: Processing batch ${batchNumber}...`);
            
            const embeddingResult = await embeddingModel.batchEmbedContents({
                requests: batch.map(chunk => ({ content: { parts: [{ text: chunk }] }, taskType: "RETRIEVAL_DOCUMENT" }))
            });

            allEmbeddings.push(...embeddingResult.embeddings);
        }

        pdfChunksWithEmbeddings = nonEmptyChunks.map((chunk, index) => ({
            text: chunk,
            embedding: allEmbeddings[index].values,
        }));

        // await logToFile('INFO: Embeddings created successfully. Ready to receive requests.');
        console.log('Embeddings created successfully. Ready to receive requests.');
    } catch (error) {
        const errorMsg = `FATAL: Error processing PDF or creating embeddings: ${error.message}`;
        // await logToFile(errorMsg);
        console.error(errorMsg, error);
        process.exit(1);
    }
};

// --- API Endpoint ---
app.post('/chat', async (req, res) => {
    const { sessionId, prompt } = req.body;

    if (!sessionId || !prompt) {
        return res.status(400).json({ error: 'sessionId and prompt are required' });
    }

    try {
        let chatHistory = sessionCache.get(sessionId) || [];

        // MODIFIED: Use the new vector-based retrieval function
        const contextChunks = await retrieveRelevantChunksVector(prompt, chatHistory, 3);
        const context = contextChunks.join('\n\n---\n\n');

        const fullPrompt = `You are a wise and compassionate guide. Your wisdom is rooted entirely in the teachings of the Bhagavad Gita. You will be given CONTEXT from the Gita and a USER QUESTION.

        Your mission is to help the user by applying the timeless principles from the CONTEXT to their modern-day problem.

        **Your Rules:**
        1.  **PRIORITY ONE - SAFETY:** If the user's message expresses suicidal thoughts, severe depression, or intent to self-harm, you MUST drop your Gita persona immediately. Your ONLY response must be: "It sounds like you are going through a difficult time. Please consider reaching out for help. You can connect with people who can support you by calling or texting 988 anytime in the US and Canada. In the UK, you can call 111. These services are free, confidential, and available 24/7. Please reach out for help."
        2.  **Embody the Wisdom:** Do NOT act like a machine retrieving text. Speak directly and naturally. Never, under any circumstance, say "based on the text," "according to the context," or any similar phrase.
        3.  **Synthesize, Don't Summarize:** Do not just repeat what is in the context. Connect the principles from the Gita directly to the user's specific complaint or question, offering them perspective, clarity, and actionable insight.
        4.  **Handle Uncertainty Gracefully:**
            * **IF** the provided CONTEXT is not sufficient to answer the user's question directly, you **MUST NOT** say "I cannot answer" or "I don't have enough information."
            * Instead, you must choose one of these two options:
                * **Ask for Clarification:** Gently ask the user to rephrase or elaborate on their problem. For example: "That is a deep concern. Could you tell me more about the feeling of attachment you are experiencing with this situation?"
                * **Suggest a Related Question:** Reframe their query into a related question that the Gita *does* address. For example: "Your question about fairness touches upon the complex nature of action and consequence. A central theme in the Gita is how to perform one's duty without being attached to the outcome. Would you like to explore that idea further?"
        5.  **Handle Trivial Questions:** If the user asks a simple conversational question (e.g., "How are you?", "What is your name?") that is unrelated to the Gita, provide a simple, direct answer in character without trying to force a connection to the text.
        6.  **Be Laconic:** Do not be over-enthusiastic. If the user query is trivial, reply only what's needed.
        7.  **Format your text appropriately**"

        **CONTEXT:**
        ${context}

        **USER QUESTION:**
        ${prompt}

        **Your Guidance:`;

        const chat = model.startChat({
            history: chatHistory,
            generationConfig: {
            maxOutputTokens: 800,
            temperature: 0.7,
            topP: 0.9,
            },
        });

        const result = await chat.sendMessage(fullPrompt);
        const text = result?.response?.text?.() ?? '';
        const finalText = text.trim().length > 0 ? text : "Could you share a bit more detail so I can offer clearer guidance?";

        chatHistory.push({ role: 'user', parts: [{ text: prompt }] });
        chatHistory.push({ role: 'model', parts: [{ text: finalText }] });
        sessionCache.set(sessionId, chatHistory);

        res.json({ response: finalText });

    } catch (error) {
        console.error('Error in /chat endpoint:', error);
        res.status(500).json({ error: 'An internal server error occurred.' });
    }
});

const loadEmbeddings = async () => {
    try {
        console.log('Loading pre-computed embeddings...');
        const data = await fs.readFile('gita-embeddings.json', 'utf-8');
        pdfChunksWithEmbeddings = JSON.parse(data);
        
        if (!pdfChunksWithEmbeddings || pdfChunksWithEmbeddings.length === 0) {
           throw new Error("Embeddings file is empty or invalid.");
        }

        console.log(`✅ Embeddings loaded successfully. ${pdfChunksWithEmbeddings.length} chunks ready.`);
    } catch (error) {
        const errorMsg = `FATAL: Could not load embeddings file: ${error.message}`;
        console.error(errorMsg, error);
        process.exit(1); // Exit if embeddings can't be loaded
    }
};

// --- Server Startup ---
const startServer = async () => {
    await loadEmbeddings(); // Process the PDF before starting the server
    app.listen(PORT, () => {
        console.log(`Server is running on http://localhost:${PORT}`);
    });
};

startServer();
