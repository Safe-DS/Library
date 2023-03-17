// Based on https://github.com/bjoluc/semantic-release-config-poetry/blob/main/release.config.js

const PYPI_REPOSITORY = 'https://upload.pypi.org/legacy/';

module.exports = {
    branches: ['main'],
    plugins: [
        ['@semantic-release/commit-analyzer', { preset: 'conventionalcommits' }],
        ['@semantic-release/release-notes-generator', { preset: 'conventionalcommits' }],
        ['@semantic-release/changelog', { changelogFile: 'docs/CHANGELOG.md' }],
        [
            '@semantic-release/exec',
            {
                verifyConditionsCmd: `if [ 403 != $(curl -X POST -F ':action=file_upload' -u __token__:$PYPI_TOKEN -s -o /dev/null -w '%{http_code}' ${PYPI_REPOSITORY}) ]; then (exit 0); else (echo 'Authentication error. Please check the PYPI_TOKEN environment variable.' && exit 1); fi`,
                prepareCmd: 'poetry version ${nextRelease.version}',
                publishCmd: 'poetry publish --build --username __token__ --password $PYPI_TOKEN --no-interaction -vvv',
            },
        ],
        [
            '@semantic-release/github',
            {
                assets: [
                    { path: 'dist/*.tar.gz', label: 'sdist' },
                    { path: 'dist/*.whl', label: 'wheel' },
                ],
            },
        ],
        [
            '@semantic-release/git',
            {
                assets: ['pyproject.toml', 'docs/CHANGELOG.md'],
            },
        ],
    ],
};
