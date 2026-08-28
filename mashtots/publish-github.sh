#!/usr/bin/env bash
# Публикация решения Mashtots на GitHub.
# Запускайте локально, где у вас есть доступ к GitHub (не в Cloud Agent с ограниченным токеном).
#
# Использование:
#   chmod +x publish-github.sh
#   ./publish-github.sh merge-pr          # смержить PR #1 в main (рекомендуется)
#   ./publish-github.sh push-branch       # только запушить ветку с изменениями
#   ./publish-github.sh new-repo          # создать отдельный репозиторий mashtots-letters-cnn
#   ./publish-github.sh kaggle-push       # отправить ноутбук в Kaggle

set -euo pipefail

GITHUB_USER="${GITHUB_USER:-AnnaKazarian13}"
EXISTING_REPO="${EXISTING_REPO:-clustering-physical-activity}"
PR_NUMBER="${PR_NUMBER:-1}"
BRANCH="${BRANCH:-cursor/fix-mashtots-letters-cnn-0020}"
NEW_REPO="${NEW_REPO:-mashtots-letters-cnn}"
KAGGLE_KERNEL_ID="${KAGGLE_KERNEL_ID:-annakazarian/mashtots-letters-cnn}"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Ошибка: не найдена команда '$1'. Установите её и повторите."
    exit 1
  }
}

check_gh_auth() {
  require_cmd gh
  if ! gh auth status -h github.com >/dev/null 2>&1; then
    echo "GitHub CLI не авторизован. Выполните:"
    echo "  gh auth login"
    exit 1
  fi
}

merge_pr() {
  check_gh_auth
  echo "==> Смержить PR #${PR_NUMBER} в ${GITHUB_USER}/${EXISTING_REPO}"
  gh pr view "${PR_NUMBER}" \
    --repo "${GITHUB_USER}/${EXISTING_REPO}" \
    --json url,title,state,mergeable

  gh pr merge "${PR_NUMBER}" \
    --repo "${GITHUB_USER}/${EXISTING_REPO}" \
    --merge \
    --delete-branch

  echo "Готово. PR: https://github.com/${GITHUB_USER}/${EXISTING_REPO}/pull/${PR_NUMBER}"
}

push_branch() {
  require_cmd git
  check_gh_auth

  local repo_url="https://github.com/${GITHUB_USER}/${EXISTING_REPO}.git"
  local workdir
  workdir="$(mktemp -d)"

  echo "==> Клонировать ${repo_url}"
  git clone "${repo_url}" "${workdir}/repo"
  cd "${workdir}/repo"

  git fetch origin "${BRANCH}" || true
  if git show-ref --verify --quiet "refs/remotes/origin/${BRANCH}"; then
    git checkout "${BRANCH}"
  else
    git checkout -b "${BRANCH}"
  fi

  echo "==> Скопировать папку mashtots/ (если запускаете из родительского каталога)"
  if [[ -d "${OLDPWD}/mashtots" ]]; then
    rsync -a --delete "${OLDPWD}/mashtots/" "./mashtots/"
    git add mashtots/ README.md .gitignore requirements.txt 2>/dev/null || git add mashtots/
    if ! git diff --cached --quiet; then
      git commit -m "Mashtots: рабочий, учебный и локальный ноутбуки"
    fi
  fi

  git push -u origin "${BRANCH}"
  echo "Ветка запушена: ${BRANCH}"
  echo "Создайте PR: https://github.com/${GITHUB_USER}/${EXISTING_REPO}/compare/main...${BRANCH}"
}

new_repo() {
  check_gh_auth
  require_cmd git

  local repo_url="https://github.com/${GITHUB_USER}/${NEW_REPO}.git"
  local workdir
  workdir="$(mktemp -d)"

  echo "==> Создать репозиторий ${GITHUB_USER}/${NEW_REPO}"
  gh repo create "${GITHUB_USER}/${NEW_REPO}" \
    --public \
    --description "Mashtots Dataset: CNN для классификации 78 армянских рукописных букв (Kaggle)" \
    --clone "${workdir}/${NEW_REPO}" \
    || echo "Репозиторий уже существует — продолжаем с клоном"

  if [[ ! -d "${workdir}/${NEW_REPO}/.git" ]]; then
    git clone "${repo_url}" "${workdir}/${NEW_REPO}"
  fi

  cd "${workdir}/${NEW_REPO}"

  local source_dir="${OLDPWD}"
  if [[ ! -f "${source_dir}/mashtots_kaggle.ipynb" && -d "${OLDPWD}/mashtots" ]]; then
    source_dir="${OLDPWD}/mashtots"
  fi

  rsync -a \
    --exclude publish-github.sh \
    "${source_dir}/" ./

  cat > README.md <<'EOF'
# Mashtots — классификация армянских рукописных букв

Решение для [Mashtots Dataset](https://www.kaggle.com/competitions/mashtots-dataset)
и [Mashtots Dataset v2](https://www.kaggle.com/competitions/mashtots-dataset-v2).

| Файл | Назначение |
|---|---|
| `mashtots_kaggle.ipynb` | запуск в Kaggle, GPU, TTA, submission.csv |
| `mashtots_tutorial.ipynb` | учебный разбор с комментариями к каждой строке |
| `mashtoc_letters.ipynb` | локальный прогон и разбор ошибок |
| `kernel-metadata.json` | `kaggle kernels push -p .` |

Подробности — в комментариях внутри ноутбуков и в `requirements.txt`.
EOF

  git add .
  git commit -m "Initial commit: Mashtots letters CNN (Kaggle)" || true
  git branch -M main
  git push -u origin main

  echo "Готово: https://github.com/${GITHUB_USER}/${NEW_REPO}"
}

kaggle_push() {
  require_cmd kaggle
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  if [[ ! -f "${HOME}/.kaggle/kaggle.json" ]]; then
    echo "Ошибка: нет ~/.kaggle/kaggle.json"
    echo "Скачайте API token на https://www.kaggle.com/settings"
    exit 1
  fi

  cd "${script_dir}"
  echo "==> kaggle kernels push (${KAGGLE_KERNEL_ID})"
  kaggle kernels push -p .
  echo "Готово: https://www.kaggle.com/code/${KAGGLE_KERNEL_ID}"
}

usage() {
  cat <<EOF
Использование: $0 <команда>

Команды:
  merge-pr       Смержить PR #${PR_NUMBER} в ${EXISTING_REPO} (код уже на GitHub)
  push-branch    Запушить ветку ${BRANCH}
  new-repo       Создать отдельный репозиторий ${NEW_REPO}
  kaggle-push    Отправить ноутбук в Kaggle

Переменные окружения:
  GITHUB_USER, EXISTING_REPO, PR_NUMBER, BRANCH, NEW_REPO, KAGGLE_KERNEL_ID
EOF
}

main() {
  case "${1:-}" in
    merge-pr) merge_pr ;;
    push-branch) push_branch ;;
    new-repo) new_repo ;;
    kaggle-push) kaggle_push ;;
    *) usage; exit 1 ;;
  esac
}

main "$@"
