

import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

import {
  explainSelection,
  explainTerm,
  getHealth,
  getTerm,
  uploadPdf,
} from "./api/wordlensApi";

import Header from "./components/Header";
import PdfReader from "./components/PdfReader";
import FloatingExplanation from "./components/FloatingExplanation";

export default function App() {
  const [healthState, setHealthState] = useState("checking");
  const [health, setHealth] = useState(null);
  const [file, setFile] = useState(null);
  const [documentData, setDocumentData] = useState(null);
  const [detectedTerms, setDetectedTerms] = useState([]);
  const [hoveredTerm, setHoveredTerm] = useState(null);
  const [interactionMode, setInteractionMode] = useState(null);
  const [anchorPosition, setAnchorPosition] = useState({ x: 20, y: 80 });
  const [explanation, setExplanation] = useState(null);
  const [explanationLoading, setExplanationLoading] = useState(false);
  const [explanationError, setExplanationError] = useState("");
  const [uploadLoading, setUploadLoading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [selectionLoading, setSelectionLoading] = useState(false);

  const hoverAbortRef = useRef(null);
  const selectionAbortRef = useRef(null);
  const hoverTimerRef = useRef(null);
  const hideTimerRef = useRef(null);

  useEffect(() => {
    const controller = new AbortController();
    async function checkHealth() {
      try {
        const result = await getHealth(controller.signal);
        const data = result.data;
        setHealth(data);
        setHealthState(data?.status === "ready" ? "ready" : "unavailable");
      } catch (error) {
        if (error?.name === "AbortError") return;
        setHealthState("unavailable");
        setHealth({ message: "The WordLens backend is unavailable. Start FastAPI on port 8000." });
      }
    }
    checkHealth();
    return () => controller.abort();
  }, []);

  const clearExplanation = useCallback(() => {
    hoverAbortRef.current?.abort();
    selectionAbortRef.current?.abort();
    if (hoverTimerRef.current) clearTimeout(hoverTimerRef.current);
    if (hideTimerRef.current) clearTimeout(hideTimerRef.current);
    hoverTimerRef.current = null;
    hideTimerRef.current = null;
    setHoveredTerm(null);
    setInteractionMode(null);
    setExplanation(null);
    setExplanationError("");
    setExplanationLoading(false);
    setSelectionLoading(false);
  }, []);

  async function handleUpload({ file: selectedFile, error }) {
    setUploadError(error || "");
    if (error || !selectedFile) return;
    setUploadLoading(true);
    setUploadError("");
    clearExplanation();
    setFile(null);
    setDocumentData(null);
    setDetectedTerms([]);
    try {
      const result = await uploadPdf(selectedFile);
      setFile(selectedFile);
      setDocumentData(result.data);
      setDetectedTerms(result.data?.detected_terms || []);
    } catch (requestError) {
      setUploadError(requestError?.message || "The PDF could not be uploaded.");
    } finally {
      setUploadLoading(false);
    }
  }

  const setAnchorFromRect = useCallback((rect) => {
    if (!rect) return;
    setAnchorPosition({ x: rect.right + 14, y: rect.top });
  }, []);

  const showCanonicalThenRag = useCallback(async ({ term, context, controller }) => {
    // First preserve the previously working behavior: verified terminology is
    // displayed immediately. RAG is an enhancement, not a replacement for it.
    const canonicalResult = await getTerm(term, controller.signal);
    if (controller.signal.aborted) return;

    const canonical = canonicalResult.data;
    setExplanation({
      ...canonical,
      requested_term: term,
      resolved_term: canonical.term,
      canonical_definition: canonical.definition,
      simplified_definition: canonical.simplified_definition,
      grounding_status: "canonical",
      generation_method: "verified_knowledge",
      generated_explanation: null,
      sources: canonical.source
        ? [{ source: canonical.source, source_url: canonical.source_url }]
        : [],
    });

    try {
      const ragResult = await explainTerm(
        { term: canonical.term, context: context || "", top_k: 5 },
        controller.signal,
      );
      if (controller.signal.aborted) return;
      const rag = ragResult.data;

      // Never replace a good canonical meaning with an abstention.
      if (rag?.grounding_status === "grounded" && rag.generated_explanation) {
        setExplanation((previous) => ({
          ...previous,
          ...rag,
          requested_term: term,
          resolved_term: canonical.term,
          canonical_definition: canonical.definition,
          simplified_definition: canonical.simplified_definition,
          sources: rag.sources?.length ? rag.sources : previous?.sources || [],
        }));
      }
    } catch (ragError) {
      if (ragError?.name === "AbortError") return;
      // Canonical meaning is already visible, so RAG failure is non-fatal.
    }
  }, []);

  const handleTermHover = useCallback(
    ({ term, rect }) => {
      if (!term?.term || healthState !== "ready") return;
      if (interactionMode === "selection") return;
      if (hideTimerRef.current) clearTimeout(hideTimerRef.current);
      if (hoverTimerRef.current) clearTimeout(hoverTimerRef.current);
      hoverAbortRef.current?.abort();
      setAnchorFromRect(rect);
      setInteractionMode("hover");
      setHoveredTerm(term);
      setExplanation(null);
      setExplanationError("");
      setExplanationLoading(true);

      hoverTimerRef.current = setTimeout(async () => {
        const controller = new AbortController();
        hoverAbortRef.current = controller;
        try {
          await showCanonicalThenRag({
            term: term.term,
            context: term.sentence || "",
            controller,
          });
        } catch (error) {
          if (error?.name === "AbortError") return;
          setExplanationError(error?.message || "Unable to retrieve the meaning.");
        } finally {
          if (!controller.signal.aborted) setExplanationLoading(false);
        }
      }, 120);
    },
    [healthState, interactionMode, setAnchorFromRect, showCanonicalThenRag],
  );

  const handleTermLeave = useCallback(() => {
    if (interactionMode === "selection") return;
    if (hoverTimerRef.current) clearTimeout(hoverTimerRef.current);
    if (hideTimerRef.current) clearTimeout(hideTimerRef.current);
    hideTimerRef.current = setTimeout(() => {
      hoverAbortRef.current?.abort();
      setHoveredTerm(null);
      setInteractionMode(null);
      setExplanation(null);
      setExplanationError("");
      setExplanationLoading(false);
    }, 300);
  }, [interactionMode]);

  const handleSelectionChange = useCallback(
    async (selectionPayload) => {
      const selectedText = typeof selectionPayload === "string"
        ? selectionPayload.trim()
        : selectionPayload?.text?.trim() || "";
      const context = typeof selectionPayload === "object"
        ? selectionPayload?.context?.trim() || selectedText
        : selectedText;
      const rect = typeof selectionPayload === "object" ? selectionPayload?.rect : null;
      if (selectedText.length < 2 || healthState !== "ready") return;

      setInteractionMode("selection");
      hoverAbortRef.current?.abort();
      if (hoverTimerRef.current) clearTimeout(hoverTimerRef.current);
      selectionAbortRef.current?.abort();
      const controller = new AbortController();
      selectionAbortRef.current = controller;
      setSelectionLoading(true);
      setExplanationLoading(false);
      setExplanation(null);
      setExplanationError("");
      setHoveredTerm({ term: selectedText });
      setAnchorFromRect(rect);

      try {
        // Exact selection first: this preserves canonical definitions for known terms.
        try {
          await showCanonicalThenRag({ term: selectedText, context, controller });
          if (controller.signal.aborted) return;
          return;
        } catch (canonicalError) {
          if (canonicalError?.name === "AbortError") return;
          if (canonicalError?.status !== 404) throw canonicalError;
        }

        const selectionResult = await explainSelection(
          { text: selectedText, context, top_k_per_term: 3 },
          controller.signal,
        );
        if (controller.signal.aborted) return;

        const response = selectionResult.data;
        const explanations = Array.isArray(response?.explanations) ? response.explanations : [];
        const usable = explanations.find((item) => item?.grounding_status === "grounded");

        if (usable) {
          setExplanation(usable);
          setHoveredTerm({ term: usable.term || selectedText });
        } else {
          setExplanation({
            requested_term: selectedText,
            grounding_status: "insufficient_evidence",
            generation_method: "none",
            generated_explanation: null,
            canonical_definition: null,
            simplified_definition: null,
            warnings: response?.warnings || ["No grounded technical explanation was found for this selection."],
            sources: [],
          });
        }
      } catch (error) {
        if (error?.name === "AbortError") return;
        setExplanationError(error?.message || "Unable to explain the selected text.");
      } finally {
        if (!controller.signal.aborted) setSelectionLoading(false);
      }
    },
    [healthState, setAnchorFromRect, showCanonicalThenRag],
  );

  const handleSelectionClear = useCallback(() => {
    selectionAbortRef.current?.abort();
    if (hoverTimerRef.current) clearTimeout(hoverTimerRef.current);
    if (interactionMode !== "selection") return;
    setInteractionMode(null);
    setHoveredTerm(null);
    setExplanation(null);
    setExplanationError("");
    setExplanationLoading(false);
    setSelectionLoading(false);
  }, [interactionMode]);

  useEffect(() => () => {
    hoverAbortRef.current?.abort();
    selectionAbortRef.current?.abort();
    if (hoverTimerRef.current) clearTimeout(hoverTimerRef.current);
    if (hideTimerRef.current) clearTimeout(hideTimerRef.current);
  }, []);

  return (
    <div className="app">
      <Header healthState={healthState} health={health} onUpload={handleUpload} uploadLoading={uploadLoading} />
      {uploadError && <div className="global-alert error-alert" role="alert">{uploadError}</div>}
      {documentData?.warnings?.length > 0 && (
        <div className="document-warning">
          {documentData.warnings.map((warning, index) => <p key={index}>{warning}</p>)}
        </div>
      )}
      <main className="reader-workspace">
        <PdfReader
          file={file}
          detectedTerms={detectedTerms}
          onSelectionChange={handleSelectionChange}
          onSelectionClear={handleSelectionClear}
          onTermHover={handleTermHover}
          onTermLeave={handleTermLeave}
        />
        <FloatingExplanation
          state={interactionMode === "selection" || hoveredTerm ? "visible" : "hidden"}
          term={hoveredTerm}
          explanation={explanation}
          loading={explanationLoading || selectionLoading}
          error={explanationError}
          position={anchorPosition}
        />
      </main>
    </div>
  );
}
