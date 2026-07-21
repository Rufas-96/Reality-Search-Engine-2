import dotenv from "dotenv";
dotenv.config();
import express from "express";
import path from "path";
import fs from "fs";
import multer from "multer";
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const { PDFParse } = require("pdf-parse");
import cors from "cors";
import { GoogleGenAI } from "@google/genai";
import { createServer as createViteServer } from "vite";
import { VectorStore } from "./server/vectorStore.js";

// Make sure ES modules work correctly with local imports
// Note that when compiling with tsx we can import .ts or .js, but to support the esbuild bundle output let's import locally.
const vectorStore = new VectorStore();

const app = express();
const PORT = 3000;

app.use(cors());
app.use(express.json());

// Set up directories
const uploadsDir = path.join(process.cwd(), "uploads");
const dbDir = path.join(process.cwd(), "vector_db");
fs.mkdirSync(uploadsDir, { recursive: true });
fs.mkdirSync(dbDir, { recursive: true });

const memoryPath = path.join(dbDir, "memory.json");

// Multer storage
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadsDir);
  },
  filename: (req, file, cb) => {
    // Avoid path traversal and strip non-alphanumeric chars (keep dots, dashes, underscores)
    const sanitized = file.originalname.replace(/[^a-zA-Z0-9.\-_]/g, "_");
    cb(null, `${Date.now()}_${sanitized}`);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB limit
  fileFilter: (req, file, cb) => {
    if (file.mimetype === "application/pdf" || file.originalname.toLowerCase().endsWith(".pdf")) {
      cb(null, true);
    } else {
      cb(new Error("Only PDF files are supported initially."));
    }
  },
});

// Load Memory helper
function loadMemory() {
  try {
    if (fs.existsSync(memoryPath)) {
      return JSON.parse(fs.readFileSync(memoryPath, "utf-8"));
    }
  } catch (err) {
    console.error("Error loading memory:", err);
  }
  return [];
}

// Save Memory helper
function saveMemory(data: any) {
  try {
    fs.writeFileSync(memoryPath, JSON.stringify(data, null, 2), "utf-8");
  } catch (err) {
    console.error("Error saving memory:", err);
  }
}

// Initialize Gemini for general chat
const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY || "",
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    }
  }
});

// ==========================================
// API ROUTES
// ==========================================

/**
 * 1. Upload API: POST /api/upload
 */
app.post("/api/upload", upload.single("pdf"), async (req, res): Promise<any> => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No PDF file uploaded" });
    }

    const filePath = req.file.path;
    const originalName = req.file.originalname;
    const finalFilename = req.file.filename;

    console.log(`Uploaded file: ${originalName} saved as ${finalFilename}. Parsing PDF...`);

    // Extract text from PDF
    const fileBuffer = fs.readFileSync(filePath);
    const parser = new PDFParse({ data: fileBuffer });
    let extractedText = "";
    try {
      const parsedPdf = await parser.getText();
      extractedText = parsedPdf.text || "";
    } finally {
      await parser.destroy().catch(() => {});
    }

    if (!extractedText.trim()) {
      // Remove empty or un-extractable file
      try { fs.unlinkSync(filePath); } catch (_) {}
      return res.status(400).json({ error: "Failed to extract readable text from PDF. It may be scanned or empty." });
    }

    // Index the parsed text chunks in our vector store
    const chunkCount = await vectorStore.indexDocument(finalFilename, extractedText);

    res.json({
      filename: finalFilename,
      originalName,
      chunks: chunkCount,
      message: "Document indexed successfully"
    });
  } catch (err: any) {
    console.error("Error handling PDF upload:", err);
    res.status(500).json({ error: err.message || "Failed to process PDF upload." });
  }
});

/**
 * 2. Chat API: POST /api/chat
 */
app.post("/api/chat", async (req, res): Promise<any> => {
  try {
    const { question } = req.body;
    if (!question) {
      return res.status(400).json({ error: "Question is required" });
    }

    console.log(`Received question: "${question}". Executing vector search...`);

    // Perform similarity search
    const results = await vectorStore.search(question, 4);

    // Build context
    const context = results
      .map(r => `[Source Document: ${r.chunk.filename}, Chunk: ${r.chunk.chunkIndex}]\n${r.chunk.text}`)
      .join("\n\n---\n\n");

    const systemInstruction = `
You are the "Reality Search Engine" assistant, a highly precise AI chatbot that answers questions based STRICTLY on the user's uploaded documents.

CRITICAL DIRECTIVES:
1. Base your answer strictly and exclusively on the provided Document Context below.
2. If the information to answer the question cannot be found or inferred from the provided context, you MUST respond EXACTLY with:
   "I couldn't find information related to your question in the uploaded documents."
3. Do not assume, hallucinate, or reference general internet/world knowledge. If it is not in the context, say the exact phrase above. No pleasantries, no guessing.
4. Keep the answer professional, concise, and structured. Always cite sources when referencing facts.
`;

    const prompt = `
Document Context:
${context || "No context documents are uploaded or indexed yet."}

User Question:
${question}
`;

    console.log(`Sending context and instruction to Gemini...`);
    
    let geminiRes;
    let modelUsed = "gemini-3.5-flash";
    try {
      geminiRes = await ai.models.generateContent({
        model: "gemini-3.5-flash",
        contents: prompt,
        config: {
          systemInstruction,
          temperature: 0.1, // low temperature for precise RAG behavior
        }
      });
    } catch (primaryErr: any) {
      console.warn(`Primary model gemini-3.5-flash failed or experienced high demand. Trying fallback model gemini-3.1-flash-lite... Error:`, primaryErr?.message || primaryErr);
      try {
        modelUsed = "gemini-3.1-flash-lite";
        geminiRes = await ai.models.generateContent({
          model: "gemini-3.1-flash-lite",
          contents: prompt,
          config: {
            systemInstruction,
            temperature: 0.1,
          }
        });
      } catch (fallbackErr: any) {
        console.error("Both primary and fallback models failed:", fallbackErr);
        throw new Error(`AI model service is temporarily unavailable due to high demand. Please try again in a few moments.`);
      }
    }

    const answer = geminiRes.text || "I couldn't find information related to your question in the uploaded documents.";

    // Map sources for the response
    const sources = results.map(r => ({
      filename: r.chunk.filename,
      chunk: r.chunk.chunkIndex,
      text: r.chunk.text.slice(0, 150) + "..."
    }));

    res.json({
      answer,
      sources
    });
  } catch (err: any) {
    console.error("Error handling chat:", err);
    res.status(500).json({ error: err.message || "An error occurred during retrieval." });
  }
});

/**
 * 3. Documents API: GET /api/documents
 */
app.get("/api/documents", (req, res) => {
  try {
    const docs = vectorStore.getDocuments();
    res.json(docs);
  } catch (err: any) {
    res.status(500).json({ error: "Failed to retrieve documents list" });
  }
});

/**
 * DELETE /api/documents/:filename
 */
app.delete("/api/documents/:filename", (req, res) => {
  try {
    const { filename } = req.params;
    const deleted = vectorStore.deleteDocument(filename);
    res.json({ success: deleted, message: `Document '${filename}' deleted successfully.` });
  } catch (err: any) {
    res.status(500).json({ error: "Failed to delete document" });
  }
});

/**
 * 4. Memory API: POST /api/memory/save
 */
app.post("/api/memory/save", (req, res) => {
  try {
    const { conversations } = req.body;
    saveMemory(conversations || []);
    res.json({ success: true, message: "Chat memory saved successfully" });
  } catch (err) {
    res.status(500).json({ error: "Failed to save memory" });
  }
});

/**
 * GET /api/memory
 */
app.get("/api/memory", (req, res) => {
  try {
    const memory = loadMemory();
    res.json(memory);
  } catch (err) {
    res.status(500).json({ error: "Failed to load memory" });
  }
});

/**
 * DELETE /api/memory
 */
app.delete("/api/memory", (req, res) => {
  try {
    saveMemory([]);
    res.json({ success: true, message: "Chat memory cleared" });
  } catch (err) {
    res.status(500).json({ error: "Failed to clear memory" });
  }
});

// Health check endpoint
app.get("/api/health", (req, res) => {
  res.json({ status: "healthy", timestamp: new Date() });
});

// ==========================================
// VITE OR STATIC FILE SERVING
// ==========================================
async function start() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Reality Search Engine Server running on http://localhost:${PORT}`);
  });
}

start();
