use anyhow::{Result, anyhow};
use globset::{Glob, GlobMatcher};
use std::path::Path;

#[derive(Default)]
pub struct WorkbookFilter {
    slug_prefix: Option<String>,
    folder: Option<String>,
    path_glob: Option<GlobMatcher>,
}

impl WorkbookFilter {
    pub fn new(
        slug_prefix: Option<String>,
        folder: Option<String>,
        path_glob: Option<String>,
    ) -> Result<Self> {
        let matcher = if let Some(glob) = path_glob {
            Some(
                Glob::new(&glob)
                    .map_err(|err| anyhow!("invalid glob pattern {glob}: {err}"))?
                    .compile_matcher(),
            )
        } else {
            None
        };

        Ok(Self {
            slug_prefix: slug_prefix.map(|s| s.to_ascii_lowercase()),
            folder: folder.map(|s| s.to_ascii_lowercase()),
            path_glob: matcher,
        })
    }

    pub fn matches(&self, slug: &str, folder: Option<&str>, path: &Path) -> bool {
        if let Some(prefix) = &self.slug_prefix
            && !slug.to_ascii_lowercase().starts_with(prefix)
        {
            return false;
        }

        if let Some(expected_folder) = &self.folder {
            match folder.map(|f| f.to_ascii_lowercase()) {
                Some(actual) if &actual == expected_folder => {}
                _ => return false,
            }
        }

        if let Some(glob) = &self.path_glob
            && !glob.is_match(path)
        {
            return false;
        }

        true
    }
}
