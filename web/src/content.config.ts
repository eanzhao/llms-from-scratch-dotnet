import { defineCollection, z } from "astro:content";

const docs = defineCollection({
  schema: z.object({
    title: z.string(),
    order: z.number(),
    chapter: z.number().optional(),
    section: z.enum(["guide", "chapter", "appendix", "reference"]).optional(),
    summary: z.string(),
    status: z.enum(["planned", "in-progress", "done"]).default("planned"),
    tags: z.array(z.string()).default([])
  })
});

export const collections = { docs };
