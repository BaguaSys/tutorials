#!/bin/bash

set -euo pipefail

wget 'https://github.com/rust-lang/mdBook/releases/download/v0.4.10/mdbook-v0.4.10-x86_64-unknown-linux-gnu.tar.gz'
tar xzf mdbook-v0.4.10-x86_64-unknown-linux-gnu.tar.gz

wget 'https://github.com/lzanini/mdbook-katex/releases/download/v0.2.10/mdbook-katex-v0.2.10-x86_64-unknown-linux-musl.tar.gz'
tar xzf 'mdbook-katex-v0.2.10-x86_64-unknown-linux-musl.tar.gz'
mv target/x86_64-unknown-linux-musl/release/mdbook-katex .

wget 'https://github.com/badboy/mdbook-open-on-gh/releases/download/2.0.1/mdbook-open-on-gh-2.0.1-x86_64-unknown-linux-musl.tar.gz'
tar xzf 'mdbook-open-on-gh-2.0.1-x86_64-unknown-linux-musl.tar.gz'

env PATH=$PWD:$PATH mdbook build
