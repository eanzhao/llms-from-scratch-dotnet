export function withBasePath(path: string): string {
  const basePath = import.meta.env.BASE_URL;
  const normalizedBase = basePath.endsWith("/") ? basePath : `${basePath}/`;
  const normalizedPath = path.startsWith("/") ? path.slice(1) : path;

  return `${normalizedBase}${normalizedPath}`.replace(/\/{2,}/g, "/");
}
