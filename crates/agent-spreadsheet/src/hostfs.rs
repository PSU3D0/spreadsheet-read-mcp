//! Host temp-file staging, with a compile-compatible no-host fallback.
//!
//! Several canonical code paths (fork lifecycle, staged bundles, session event
//! replay) stage work through a host temporary file. Those paths are part of
//! the canonical registry surface, so they must keep compiling for every
//! target, but the wasm32 byte/session adapter has no host filesystem and never
//! reaches them: per surface rule 4 it binds workbook bytes and exports bytes.
//!
//! With the `native-fs` feature this module is a thin re-export of `tempfile`.
//! Without it, the same API exists but every constructor fails, so `tempfile`
//! (and its `rustix`/`getrandom` chain) is never compiled into the WASM bundle.

#[cfg(feature = "native-fs")]
pub use tempfile::{Builder, NamedTempFile, PersistError, TempDir, tempdir};

#[cfg(not(feature = "native-fs"))]
mod unavailable {
    use std::io;
    use std::path::{Path, PathBuf};

    fn no_host_filesystem() -> io::Error {
        io::Error::new(
            io::ErrorKind::Unsupported,
            "this build has no host filesystem; rebuild with the `native-fs` feature",
        )
    }

    /// Stand-in for [`tempfile::TempDir`]. Never constructible here.
    #[derive(Debug)]
    pub struct TempDir(std::convert::Infallible);

    impl TempDir {
        pub fn path(&self) -> &Path {
            match self.0 {}
        }
    }

    /// Stand-in for [`tempfile::NamedTempFile`]. Never constructible here.
    #[derive(Debug)]
    pub struct NamedTempFile(std::convert::Infallible);

    impl NamedTempFile {
        pub fn new_in<P: AsRef<Path>>(_dir: P) -> io::Result<Self> {
            Err(no_host_filesystem())
        }

        pub fn path(&self) -> &Path {
            match self.0 {}
        }

        pub fn keep(self) -> Result<(std::fs::File, PathBuf), io::Error> {
            match self.0 {}
        }

        pub fn as_file(&self) -> &std::fs::File {
            match self.0 {}
        }

        pub fn persist<P: AsRef<Path>>(self, _new_path: P) -> Result<std::fs::File, PersistError> {
            match self.0 {}
        }

        pub fn persist_noclobber<P: AsRef<Path>>(
            self,
            _new_path: P,
        ) -> Result<std::fs::File, PersistError> {
            match self.0 {}
        }
    }

    impl io::Write for NamedTempFile {
        fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
            match self.0 {}
        }

        fn flush(&mut self) -> io::Result<()> {
            match self.0 {}
        }
    }

    /// Stand-in for [`tempfile::PersistError`].
    #[derive(Debug)]
    pub struct PersistError {
        pub error: io::Error,
        pub file: NamedTempFile,
    }

    /// Stand-in for [`tempfile::Builder`]. Builds nothing.
    #[derive(Debug, Default)]
    pub struct Builder;

    impl Builder {
        pub fn new() -> Self {
            Self
        }

        pub fn prefix<S: AsRef<str> + ?Sized>(self, _prefix: &S) -> Self {
            self
        }

        pub fn suffix<S: AsRef<str> + ?Sized>(self, _suffix: &S) -> Self {
            self
        }

        pub fn tempfile(self) -> io::Result<NamedTempFile> {
            Err(no_host_filesystem())
        }

        pub fn tempfile_in<P: AsRef<Path>>(self, _dir: P) -> io::Result<NamedTempFile> {
            Err(no_host_filesystem())
        }
    }

    pub fn tempdir() -> io::Result<TempDir> {
        Err(no_host_filesystem())
    }
}

#[cfg(not(feature = "native-fs"))]
pub use unavailable::{Builder, NamedTempFile, PersistError, TempDir, tempdir};
