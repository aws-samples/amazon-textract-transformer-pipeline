// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
/**
 * Vite plugin to remove type="module" from built HTML script tags.
 *
 * For some reason (suspected CORS-related), SMGT/A2I's <iframe> based presentation of the built
 * template won't run the main transpiled Vue app script tag (src/main.ts) if it's configured with
 * type="module". On the input/source side this is a requirement for Vite bundling... But since
 * we're packing everything to one asset anyway, we can simply replace <script type="module"> with
 * <script defer> for similar timing effect.
 *
 * This plugin post-edits built HTML to do this replacement wherever a script type="module" is
 * found.
 */
// External Dependencies:
import { IndexHtmlTransformResult, IndexHtmlTransformContext } from "vite";
import { Plugin } from "vite";

export function viteNoModule(): Plugin {
  return {
    name: "vite:nomodule",
    transformIndexHtml: {
      enforce: "post",
      transform(html: string, ctx?: IndexHtmlTransformContext): IndexHtmlTransformResult {
        // Only use this plugin during build
        if (!ctx || !ctx.bundle) return html;
        return html.replaceAll(/(<script\s[^>]*)type="module"([^>]*>)/gi, (_, g1, g2) => {
          return g1 + "defer" + g2;
        });
      },
    },
  };
}
