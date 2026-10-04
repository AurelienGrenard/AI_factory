\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{equation-card}[2026/08/02 AI Factory equation card]

\LoadClass[12pt]{article}
\RequirePackage[paperwidth=160mm,paperheight=60mm,margin=0mm]{geometry}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{lmodern}

\pagestyle{empty}
\setlength{\parindent}{0pt}

\AtBeginDocument{%
  \centering
  \vspace*{\fill}
  \begin{minipage}{152mm}
  \centering
  \fontsize{15}{19}\selectfont
}

\AtEndDocument{%
  \end{minipage}
  \vspace*{\fill}
}
