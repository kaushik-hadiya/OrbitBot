import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// [HR-LV-001] in an answer becomes a link to "#cite-HR-LV-001", which renders as a button that opens the
// source document. Only doc_ids the API confirmed (done.citations) are linked once the answer is final.
const CITATION = /\[([A-Z]{2,4}(?:-[A-Z0-9]+)+)\]/g;

export default function Markdown({
  text,
  citations,
  onCite,
}: {
  text: string;
  citations?: string[];
  onCite?: (docId: string) => void;
}) {
  const linked = onCite
    ? text.replace(CITATION, (m, id: string) => (!citations || citations.includes(id) ? `[${id}](#cite-${id})` : m))
    : text;
  return (
    <div className="md">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a({ href, children }) {
            if (href?.startsWith("#cite-") && onCite) {
              const id = href.slice(6);
              return (
                <button className="cite" onClick={() => onCite(id)} title={`Open ${id}`}>
                  {children}
                </button>
              );
            }
            return (
              <a href={href} target="_blank" rel="noreferrer">
                {children}
              </a>
            );
          },
        }}
      >
        {linked}
      </ReactMarkdown>
    </div>
  );
}
