\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{equation-sheet}[2026/07/31 AI Factory equation sheet]

\LoadClass[12pt]{article}
\RequirePackage[paperwidth=120mm,paperheight=90mm,margin=0mm]{geometry}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{lmodern}

\pagestyle{empty}
\setlength{\parindent}{0pt}

\AtBeginDocument{%
  \centering
  \vspace*{\fill}
  \begin{minipage}{114mm}
  \centering
  \fontsize{16}{21}\selectfont
}

\AtEndDocument{%
  \end{minipage}
  \vspace*{\fill}
}
