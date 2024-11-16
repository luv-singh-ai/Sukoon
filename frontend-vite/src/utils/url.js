const RemoveTrailingSlash = (url) => {
  return url.replace(/\/$/, '');
}

export { RemoveTrailingSlash };