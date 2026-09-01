use anyhow::Result;
#[cfg(feature = "native-fs")]
use anyhow::anyhow;
#[cfg(feature = "native-fs")]
use globset::{Glob, GlobMatcher};
use std::path::Path;

/// Path globbing is a host-filesystem concern, so the compiled matcher (and the
/// `globset` dependency behind it) only exists in `native-fs` builds. The
/// wasm32 byte/session adapter has no workspace to scan and always uses the
/// default filter.
#[derive(Default)]
pub struct WorkbookFilter {
    slug_prefix: Option<String>,
    folder: Option<String>,
    #[cfg(feature = "native-fs")]
    path_glob: Option<GlobMatcher>,
}

impl WorkbookFilter {
    pub fn new(
        slug_prefix: Option<String>,
        folder: Option<String>,
        path_glob: Option<String>,
    ) -> Result<Self> {
        #[cfg(feature = "native-fs")]
        let matcher = if let Some(glob) = path_glob {
            Some(
                Glob::new(&glob)
                    .map_err(|err| anyhow!("invalid glob pattern {glob}: {err}"))?
                    .compile_matcher(),
            )
        } else {
            None
        };
        #[cfg(not(feature = "native-fs"))]
        if path_glob.is_some() {
            anyhow::bail!("path glob filtering requires the `native-fs` feature");
        }

        Ok(Self {
            slug_prefix: slug_prefix.map(|s| s.to_ascii_lowercase()),
            folder: folder.map(|s| s.to_ascii_lowercase()),
            #[cfg(feature = "native-fs")]
            path_glob: matcher,
        })
    }

    pub fn matches(&self, slug: &str, folder: Option<&str>, path: &Path) -> bool {
        #[cfg(not(feature = "native-fs"))]
        let _ = path;

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

        #[cfg(feature = "native-fs")]
        if let Some(glob) = &self.path_glob
            && !glob.is_match(path)
        {
            return false;
        }

        true
    }
}
