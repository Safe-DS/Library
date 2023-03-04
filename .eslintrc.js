module.exports = {
    root: true,
    parserOptions: {
        tsconfigRootDir: __dirname,
        project: 'tsconfig.eslint.json',
    },
    settings: {
        jest: {
            version: 28,
        },
    },
    extends: '@lars-reimann',
};
