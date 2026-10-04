#!/usr/bin/env bash

# Render one 4:3 hero image and one wide card image from every equation.
set -euo pipefail

equation_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_dir="${equation_dir}/../static"
card_output_dir="${output_dir}/cards"
build_dir="$(mktemp -d)"

trap 'rm -rf "${build_dir}"' EXIT
cd "${equation_dir}"
mkdir -p "${card_output_dir}"

for source in ./*.tex; do
    name="$(basename "${source}" .tex)"
    pdflatex \
        -interaction=nonstopmode \
        -halt-on-error \
        -output-directory="${build_dir}" \
        "${source}" >/dev/null

    sed 's/\\documentclass{equation-sheet}/\\documentclass{equation-card}/' \
        "${source}" > "${build_dir}/${name}_card.tex"
    TEXINPUTS="${equation_dir}:" pdflatex \
        -interaction=nonstopmode \
        -halt-on-error \
        -jobname="${name}_card" \
        -output-directory="${build_dir}" \
        "${build_dir}/${name}_card.tex" >/dev/null
done

if grep -q "Overfull" "${build_dir}"/*.log; then
    printf 'LaTeX produced an overfull equation.\n' >&2
    grep -n "Overfull" "${build_dir}"/*.log >&2
    exit 1
fi

for source in ./*.tex; do
    name="$(basename "${source}" .tex)"
    pdftocairo \
        -png \
        -singlefile \
        -r 254 \
        "${build_dir}/${name}.pdf" \
        "${output_dir}/${name}"
    pdftocairo \
        -png \
        -singlefile \
        -r 254 \
        "${build_dir}/${name}_card.pdf" \
        "${card_output_dir}/${name}"
done

python3 "${equation_dir}/center_images.py" "${output_dir}" "${card_output_dir}"

printf 'Rendered hero and card equations in %s\n' "${output_dir}"
