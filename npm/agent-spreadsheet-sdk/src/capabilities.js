/**
 * @typedef {object} BackendCapabilities
 * @property {boolean} supportsDescribeWorkbook
 * @property {boolean} supportsNamedRanges
 * @property {boolean} supportsNamedRangeMutations
 * @property {boolean} supportsSheetOverview
 * @property {boolean} supportsListSheets
 * @property {boolean} supportsRangeValues
 * @property {boolean} supportsFindValue
 * @property {boolean} supportsReadTable
 * @property {boolean} supportsSheetPage
 * @property {boolean} supportsGridExport
 * @property {boolean} supportsTransformBatch
 * @property {boolean} supportsStructureBatch
 * @property {boolean} supportsReplaceInFormulas
 * @property {boolean} supportsVerification
 * @property {boolean} supportsForkLifecycle
 * @property {boolean} supportsStaging
 * @property {boolean} supportsSessionLifecycle
 * @property {boolean} supportsExportWorkbook
 */

/** @type {Readonly<BackendCapabilities>} */
const MCP_CAPABILITIES = Object.freeze({
  supportsDescribeWorkbook: true,
  supportsNamedRanges: true,
  supportsNamedRangeMutations: true,
  supportsSheetOverview: true,
  supportsListSheets: true,
  supportsRangeValues: true,
  supportsFindValue: true,
  supportsReadTable: true,
  supportsSheetPage: true,
  supportsGridExport: true,
  supportsTransformBatch: true,
  supportsStructureBatch: true,
  supportsReplaceInFormulas: true,
  supportsVerification: true,
  supportsForkLifecycle: true,
  supportsStaging: true,
  supportsSessionLifecycle: false,
  supportsExportWorkbook: false
})

/** @type {Readonly<BackendCapabilities>} */
const WASM_CAPABILITIES = Object.freeze({
  supportsDescribeWorkbook: true,
  supportsNamedRanges: true,
  supportsNamedRangeMutations: true,
  supportsSheetOverview: true,
  supportsListSheets: true,
  supportsRangeValues: true,
  supportsFindValue: true,
  supportsReadTable: true,
  supportsSheetPage: true,
  supportsGridExport: true,
  supportsTransformBatch: true,
  supportsStructureBatch: false,
  supportsReplaceInFormulas: false,
  supportsVerification: false,
  supportsForkLifecycle: false,
  supportsStaging: false,
  supportsSessionLifecycle: true,
  supportsExportWorkbook: true
})

/**
 * @param {Readonly<BackendCapabilities>} capabilities
 * @returns {Readonly<BackendCapabilities>}
 */
function freezeCapabilities(capabilities) {
  return Object.freeze({ ...capabilities })
}

module.exports = {
  MCP_CAPABILITIES,
  WASM_CAPABILITIES,
  freezeCapabilities
}
