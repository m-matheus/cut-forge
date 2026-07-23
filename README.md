# CutForge

Gerador de vídeos de música anime (estilo 7 Minutoz / Rustage / Sensei Beats). Em vez de renderizar
o vídeo final internamente, o CutForge monta uma **timeline editável** e exporta um **projeto Final
Cut Pro 7 XML** que o **Adobe Premiere Pro importa nativamente** — você edita e renderiza no Premiere.

## O que ele faz

1. **Letra & estilo** — gera um pacote de música (título, style Suno, letra) via API Anthropic.
2. **Suno (manual)** — você gera a música no [suno.ai](https://suno.ai) e coloca `track.mp3` no run.
3. **Footage** — baixa um vídeo do YouTube (AMV/edit) via yt-dlp para usar de fundo.
4. **Alinhar letra** — alinha a letra ao áudio com Whisper (timestamps por palavra).
5. **Captions** — gera legenda karaokê (`.ass`) para importar no Premiere.
6. **Thumbnail** — gera a capa (16:9) via OpenAI (Responses API / gpt-image).
7. **Metadata** — gera título/descrição/tags do YouTube via Anthropic.
8. **Exportar Premiere** — gera `premiere/project.xml`: footage em V1, música em A1, e **marcadores
   em cada linha da letra** (pra cortar no ritmo). Abra no Premiere e edite.

> **Escopo v1:** só vídeos de música, um idioma por run. Sem publish automático — o pipeline
> termina na exportação do projeto Premiere.

## Rodando (dev)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
copy .env.example .env    # preencha as chaves
python -m cutforge.ui.desktop
```

A janela do app abre com o wizard. Crie um run e vá rodando cada etapa.

## Gerando o executável (.exe)

```bash
pip install -e ".[dev]"
pyinstaller build.spec
# dist/CutForge.exe
```

## Fluxo no Premiere

1. Abra `output/{run}/premiere/project.xml` no Premiere (File → Import).
2. A sequence carrega com o footage em V1, a música em A1 e os marcadores nas linhas da letra.
3. Corte/edite o footage no ritmo (os marcadores guiam).
4. Importe `output/{run}/audio/captions.ass` como legenda, se quiser o karaokê queimado.
5. Adicione o endcard (`premiere/endcard.png`) e renderize.

## Arquitetura

`src/cutforge/` em camadas: `integrations/` (wrappers de API) → `services/` (etapas) →
`pipeline/` (orquestração idempotente) → `api/` (FastAPI) → `ui/` (janela pywebview + htmx).
Config por canal em `channels/{slug}/channel.json`.
