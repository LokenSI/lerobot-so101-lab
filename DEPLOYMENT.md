# Showcase deployment

The workflow uploads **only `site/`**. It checks out actual Git LFS payloads, validates links, rejects unresolved LFS pointers and deploys through GitHub Actions. No Git LFS credentials are embedded in the website.

Repository: https://github.com/LokenSI/lerobot-so101-lab

GitHub Pro Pages sites are public, including sites sourced from private repositories. Private website access requires an eligible GitHub Enterprise Cloud organization. See [GitHub documentation](https://docs.github.com/en/enterprise-cloud@latest/pages/getting-started-with-github-pages/changing-the-visibility-of-your-github-pages-site).

The public showcase deploys when site content changes on main, or through manual workflow dispatch.

For a new clone:

```sh
git lfs install --local
git lfs pull
python tools/verify_site.py
```
