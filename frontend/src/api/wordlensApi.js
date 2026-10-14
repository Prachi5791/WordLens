
const API_BASE_URL =
  import.meta.env.VITE_WORDLENS_API_BASE_URL ||
  "http://localhost:8000";

/**
 * Common API request helper.
 */
async function request(path, options = {}) {
  const response = await fetch(
    `${API_BASE_URL}${path}`,
    {
      ...options,

      headers: {
        Accept: "application/json",
        ...(options.headers || {}),
      },
    },
  );

  const requestId =
    response.headers.get(
      "x-request-id",
    );

  let data = null;

  try {
    data = await response.json();
  } catch {
    data = null;
  }

  if (!response.ok) {
    const error = new Error(
      data?.message ||
        data?.detail ||
        `Request failed with status ${response.status}`,
    );

    error.status =
      response.status;

    error.requestId =
      requestId ||
      data?.request_id ||
      null;

    error.details =
      data?.details || null;

    throw error;
  }

  return {
    data,
    requestId:
      requestId ||
      data?.request_id ||
      null,
  };
}

/**
 * ----------------------------------------
 * HEALTH
 * ----------------------------------------
 */

export async function getHealth(
  signal,
) {
  return request(
    "/health",
    {
      signal,
    },
  );
}

/**
 * ----------------------------------------
 * PDF UPLOAD
 * ----------------------------------------
 */

export async function uploadPdf(
  file,
  signal,
) {
  const formData =
    new FormData();

  formData.append(
    "file",
    file,
  );

  return request(
    "/upload",
    {
      method: "POST",
      body: formData,
      signal,
    },
  );
}

/**
 * ----------------------------------------
 * EXACT TERM LOOKUP
 * ----------------------------------------
 *
 * Used for:
 *
 * - hover
 * - selecting one exact
 *   technical term
 *
 * Example:
 *
 * GET /term/Generative%20AI
 */

export async function getTerm(
  term,
  signal,
) {
  const cleanTerm =
    String(term || "").trim();

  if (!cleanTerm) {
    throw new Error(
      "Term cannot be empty.",
    );
  }

  return request(
    `/term/${encodeURIComponent(
      cleanTerm,
    )}`,
    {
      signal,
    },
  );
}

/**
 * ----------------------------------------
 * CONTEXTUAL EXPLANATION
 * ----------------------------------------
 *
 * Day-3 RAG endpoint.
 *
 * Used after an exact term has
 * been resolved through /term/{term}.
 */

export async function explainTerm(
  {
    term,
    context = "",
    top_k = 5,
  },
  signal,
) {
  const cleanTerm =
    String(term || "").trim();

  if (!cleanTerm) {
    throw new Error(
      "Term cannot be empty.",
    );
  }

  return request(
    "/explain",
    {
      method: "POST",

      headers: {
        "Content-Type":
          "application/json",
      },

      body: JSON.stringify({
        term: cleanTerm,

        context:
          String(
            context || "",
          ).trim(),

        top_k,

        include_related_terms:
          false,

        include_retrieval_evidence:
          true,
      }),

      signal,
    },
  );
}

/**
 * ----------------------------------------
 * TEXT SELECTION EXPLANATION
 * ----------------------------------------
 *
 * Day-2 detector + Day-3 RAG.
 *
 * Used when the user selects:
 *
 * - a sentence
 * - multiple words
 * - text containing one or
 *   more technical terms
 */

export async function explainSelection(
  {
    text,
    context = "",
    top_k_per_term = 3,
  },
  signal,
) {
  const cleanText =
    String(text || "").trim();

  if (!cleanText) {
    throw new Error(
      "Selected text cannot be empty.",
    );
  }

  return request(
    "/explain-selection",
    {
      method: "POST",

      headers: {
        "Content-Type":
          "application/json",
      },

      body: JSON.stringify({
        text: cleanText,

        context:
          String(
            context || "",
          ).trim(),

        top_k_per_term,

        include_retrieval_evidence:
          true,
      }),

      signal,
    },
  );
}