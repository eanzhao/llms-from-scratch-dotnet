import { defineConfig } from "astro/config";

const repositoryName = process.env.GITHUB_REPOSITORY?.split("/")[1] ?? "llms-from-scratch-dotnet";
const repositoryOwner = process.env.GITHUB_REPOSITORY_OWNER ?? "eanzhao";
const isGitHubActionsBuild = process.env.GITHUB_ACTIONS === "true";

export default defineConfig({
  site: `https://${repositoryOwner}.github.io`,
  base: isGitHubActionsBuild ? `/${repositoryName}` : "/"
});
