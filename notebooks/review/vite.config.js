// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
/* global __dirname */

// Node Built-Ins:
import { resolve } from "path";
import { fileURLToPath, URL } from "url";

// External Dependencies:
import { defineConfig } from "vite";
import { viteSingleFile } from "vite-plugin-singlefile";
import vue from "@vitejs/plugin-vue";

// See reference at https://vitejs.dev/config/
export default defineConfig({
  build: {
    // Single file adjustments as per https://www.npmjs.com/package/vite-plugin-singlefile
    assetsInlineLimit: 100000000, // for vite-plugin-singlefile
    chunkSizeWarningLimit: 100000000, // for vite-plugin-singlefile
    cssCodeSplit: false, // for vite-plugin-singlefile
    reportCompressedSize: false, // Not really relevant for single-file outputs
    rollupOptions: {
      external: [
        // ---- Dependencies to exclude from build (will be fetched from CDN):
        // TRP.js is run as IIFE by a script tag in the HTML which produces a `trp` global (below):
        "amazon-textract-response-parser",
        // PDF.js entrypoints will be treated as external `paths` (below):
        "pdfjs-dist/legacy/build/pdf.mjs",
        "pdfjs-dist/legacy/web/pdf_viewer.mjs",
      ],
      // You could point the build to a different input HTML template if needed:
      input: resolve(__dirname, "index.html"),
      output: {
        format: "es", // Need to use ESM for pdf.js
        paths: {
          "pdfjs-dist/legacy/build/pdf.mjs": "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.5.136/legacy/build/pdf.mjs",
          "pdfjs-dist/legacy/web/pdf_viewer.mjs": "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.5.136/legacy/web/pdf_viewer.mjs",
        },
        globals: {
          "amazon-textract-response-parser": "trp",
        },
        inlineDynamicImports: true, // for vite-plugin-singlefile
      },
    },
  },
  plugins: [
    vue({
      template: {
        compilerOptions: {
          // Avoid default {{ }} delimiters because these will conflict with Liquid template lang.
          delimiters: ["${", "}"],
          // Declare the SageMaker Crowd HTML Elements so Vue doesn't fuss about missing them:
          isCustomElement: (tag) => tag.startsWith("crowd-") || tag.startsWith("iron-"),
        },
      },
    }),
    // Package all outputs together so we don't have to find a way to host many JS/CSS/etc assets:
    viteSingleFile(),
  ],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
});
