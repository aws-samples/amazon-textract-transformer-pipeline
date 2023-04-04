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

// Local Dependencies:
import { viteNoModule } from "./tools/vite-plugin-nomodule";

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
        // Dependencies to exclude from build because they're included via CDN in template:
        "amazon-textract-response-parser",
        "pdfjs-dist/legacy/build/pdf",
        "pdfjs-dist/legacy/web/pdf_viewer",
      ],
      // You could point the build to a different input HTML template if needed:
      input: resolve(__dirname, "index.html"),
      output: {
        format: "iife", // umd should also work
        globals: {
          // For CDN-externalized modules, what global variables does running the script set:
          "amazon-textract-response-parser": "trp",
          "pdfjs-dist/legacy/build/pdf": "pdfjsLib",
          "pdfjs-dist/legacy/web/pdf_viewer": "pdfjsViewer",
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
    // Sub script tag `type="module"` to `defer` for CORS handling in SMGT/A2I:
    viteNoModule(),
  ],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
});
