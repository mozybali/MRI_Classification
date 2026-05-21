#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
hpo.py
------
MRI siniflandirma modelleri icin Optuna TPE tabanli Bayes search girisi.

Bu dosya ortak CLI/dispatch akisini yonetir: argparser, study/storage
hazirligi, ortak validasyon yardimcisi, study artifact'lari ve gorseller.
Model-spesifik mantik iki ayri dosyaya bolunmustur:

- ``hpo_dl.py``  : ResNet/DL HPO bilesenleri
- ``hpo_xgb.py`` : XGBoost HPO bilesenleri
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from model.ayarlar import (
        HPO_KLASORU,
        RASTGELE_TOHUM,
        SL_FEATURE_CACHE_KLASORU,
        VARSAYILAN_EARLY_STOPPING_SABIR,
    )
    from model.training_runner import (
        SUPPORTED_SELECTION_METRICS,
        _selection_mode_for_metric,
        validate_training_config,
    )
    from model.sl.training_runner import validate_sl_config
    from model import hpo_dl, hpo_xgb
else:
    from .ayarlar import (
        HPO_KLASORU,
        RASTGELE_TOHUM,
        SL_FEATURE_CACHE_KLASORU,
        VARSAYILAN_EARLY_STOPPING_SABIR,
    )
    from .training_runner import (
        SUPPORTED_SELECTION_METRICS,
        _selection_mode_for_metric,
        validate_training_config,
    )
    from .sl.training_runner import validate_sl_config
    from . import hpo_dl, hpo_xgb

try:
    import optuna
except ModuleNotFoundError:
    optuna = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MRI siniflandirma icin Optuna TPE tabanli Bayes search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornekler:
  python -m model.hpo --model resnet --trials 20 --epochs 12
  python -m model.hpo --model resnet --trials 30 --metric loss --skip-final-train
  python -m model.hpo --model resnet --search-pretrained --batch-size-choices 16 32
  python -m model.hpo --model resnet --trainval-dir goruntu_isleme/cikti/trainval --test-dir goruntu_isleme/cikti/test
        """,
    )
    parser.add_argument("--model", choices=["resnet", "xgboost"], default="resnet")
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Hedef toplam trial sayisi",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Opsiyonel sure siniri (saniye)",
    )
    parser.add_argument(
        "--metric",
        choices=sorted(SUPPORTED_SELECTION_METRICS),
        default="f1",
        help="Optimize edilecek validation metrik",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optuna study adi",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Opsiyonel Optuna storage URL (ornegin sqlite:///study.db)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Arama ciktilarinin kaydedilecegi klasor",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=VARSAYILAN_EARLY_STOPPING_SABIR)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=RASTGELE_TOHUM)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--trainval-dir", type=str, default=None, help="Train+Val icin veri dizini")
    parser.add_argument("--test-dir", type=str, default=None, help="Test icin veri dizini")
    parser.add_argument(
        "--batch-size-choices",
        type=int,
        nargs="+",
        default=[16, 32, 48],
        help="Denenecek batch size adaylari",
    )
    parser.add_argument(
        "--image-size-choices",
        type=int,
        nargs="+",
        default=[160, 192, 224],
        help="Denenecek goruntu boyutlari",
    )
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--lr-max", type=float, default=5e-4)
    parser.add_argument("--weight-decay-min", type=float, default=1e-6)
    parser.add_argument("--weight-decay-max", type=float, default=1e-2)
    parser.add_argument("--scheduler-factor-min", type=float, default=0.2)
    parser.add_argument("--scheduler-factor-max", type=float, default=0.7)
    parser.add_argument("--scheduler-patience-min", type=int, default=2)
    parser.add_argument("--scheduler-patience-max", type=int, default=6)
    parser.add_argument(
        "--loss-choices",
        choices=["ce", "focal"],
        nargs="+",
        default=["ce", "focal"],
        help="Aramada kullanilacak loss adaylari",
    )
    parser.add_argument("--focal-gamma-min", type=float, default=1.0)
    parser.add_argument("--focal-gamma-max", type=float, default=4.0)
    parser.add_argument("--dropout-min", type=float, default=0.1)
    parser.add_argument("--dropout-max", type=float, default=0.6)
    parser.add_argument("--label-smoothing-min", type=float, default=0.0)
    parser.add_argument("--label-smoothing-max", type=float, default=0.15)
    parser.add_argument(
        "--hflip-p-choices",
        type=float,
        nargs="+",
        default=[0.0],
        help=(
            "Denenecek RandomHorizontalFlip olasiliklari. Beyin MR'larinda "
            "anatomik lateralite onemli oldugundan varsayilan yalnizca 0.0'dir."
        ),
    )
    parser.add_argument("--rotation-degrees-min", type=int, default=0)
    parser.add_argument("--rotation-degrees-max", type=int, default=20)
    parser.add_argument("--color-jitter-min", type=float, default=0.0)
    parser.add_argument("--color-jitter-max", type=float, default=0.2)
    parser.add_argument(
        "--search-pretrained",
        action="store_true",
        help="ResNet icin pretrained secenegini de arama uzayina ekle",
    )
    parser.add_argument(
        "--n-startup-trials",
        type=int,
        default=5,
        help="TPE sampler icin baslangic random trial sayisi",
    )
    parser.add_argument(
        "--pruner-startup-trials",
        type=int,
        default=5,
        help="Median pruner baslamadan once tamamlanacak trial sayisi",
    )
    parser.add_argument(
        "--pruner-warmup-epochs",
        type=int,
        default=3,
        help="Pruning oncesi minimum epoch sayisi",
    )
    parser.add_argument(
        "--skip-final-train",
        action="store_true",
        help="Arama sonunda en iyi parametrelerle final egitimi yapma",
    )
    parser.add_argument(
        "--verbose-trials",
        action="store_true",
        help="Her trial icin epoch loglarini goster",
    )
    parser.add_argument(
        "--feature-cache",
        type=str,
        default=str(SL_FEATURE_CACHE_KLASORU),
        help=(
            "XGBoost ozellik cache dizini (disk .npz). Bos vermek icin "
            "--no-feature-cache kullanin."
        ),
    )
    parser.add_argument(
        "--no-feature-cache",
        action="store_true",
        help="XGBoost icin disk ozellik cache'ini devre disi birak.",
    )
    parser.add_argument(
        "--xgb-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help=(
            "XGBoost device modu (sadece --model xgboost icin). "
            "'auto' torch.cuda mevcutsa GPU, aksi halde CPU secer."
        ),
    )
    parser.add_argument(
        "--xgb-n-jobs",
        type=int,
        default=None,
        help="XGBoost icin worker thread sayisi. None ise os.cpu_count().",
    )
    parser.add_argument(
        "--xgb-class-balance",
        choices=["none", "balanced"],
        default="none",
        help=(
            "XGBoost sinif dengesizligi telafisi (sadece --model xgboost icin). "
            "'none' (varsayilan): mevcut davranis. 'balanced': sklearn "
            "compute_sample_weight ile her ornege ters-frekans agirlik atanir; "
            "hem trial fit'lerinde hem final egitimde agirlik yalnizca egitim "
            "loss'una uygulanir, eval_set/erken durdurma agirliksiz birakilir."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help=(
            "Optuna study.optimize icin paralel trial sayisi. >1 yalnizca "
            "GPU/CPU baski yapmayan modlarda (orn. XGBoost CPU) anlamli; "
            "tek-GPU DL trial'lari icin 1 birakilmali."
        ),
    )
    parser.add_argument(
        "--no-hpo-plots",
        action="store_true",
        help="Optuna gorsellestirmelerini ve final dashboard'unu olusturma.",
    )
    parser.add_argument(
        "--hpo-folds",
        type=int,
        default=1,
        help=(
            "Her trial'i K-fold cross-validation ile degerlendir (>=2). 1 ise tek "
            "hold-out kullanilir (varsayilan). >1 iken epoch-bazli pruning devre "
            "disi birakilir; objective fold val metriklerinin ortalamasidir."
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _remaining_trials_to_run(requested_total_trials: int, existing_trial_count: int) -> int:
    return max(0, requested_total_trials - existing_trial_count)


def _default_study_name(args: argparse.Namespace) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{args.model}_bayes_search_{timestamp}"


def _resolve_study_dir(args: argparse.Namespace, study_name: str) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    return HPO_KLASORU / study_name


def _search_space_summary(args: argparse.Namespace) -> dict[str, Any]:
    if args.model == "xgboost":
        return hpo_xgb._search_space_summary_xgb(args)
    return hpo_dl._search_space_summary_dl(args)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def validate_search_args(args: argparse.Namespace) -> None:
    if optuna is None:
        raise ModuleNotFoundError(
            "Bayes search icin 'optuna' gerekli. "
            "Kurulum: once requirements.txt ve uygun PyTorch requirements dosyasini "
            "yukleyin, sonra .\\.venv\\Scripts\\python.exe -m pip install -e .[dev] --no-deps calistirin."
        )
    if args.trials < 1:
        raise ValueError("--trials en az 1 olmali.")
    if args.timeout is not None and args.timeout < 1:
        raise ValueError("--timeout pozitif olmali.")
    if not args.image_size_choices:
        raise ValueError("--image-size-choices bos olamaz.")
    if args.n_startup_trials < 1:
        raise ValueError("--n-startup-trials en az 1 olmali.")
    if args.pruner_startup_trials < 0:
        raise ValueError("--pruner-startup-trials negatif olamaz.")
    if args.pruner_warmup_epochs < 0:
        raise ValueError("--pruner-warmup-epochs negatif olamaz.")
    if args.hpo_folds < 1:
        raise ValueError("--hpo-folds en az 1 olmali.")
    if getattr(args, "n_jobs", 1) < 1:
        raise ValueError("--n-jobs en az 1 olmali.")

    if args.model == "xgboost":
        hpo_xgb._validate_xgb_args(args)
        return

    hpo_dl._validate_dl_args(args)


def _figure_from_plot_result(plot_result: Any) -> Any | None:
    if hasattr(plot_result, "figure"):
        return plot_result.figure
    if hasattr(plot_result, "flat"):
        for item in plot_result.flat:
            fig = _figure_from_plot_result(item)
            if fig is not None:
                return fig
    if isinstance(plot_result, (list, tuple)):
        for item in plot_result:
            fig = _figure_from_plot_result(item)
            if fig is not None:
                return fig
    return None


def _save_single_hpo_plot(study: Any, plot_func: Any, save_path: Path, label: str) -> str | None:
    import logging
    import matplotlib.pyplot as plt

    optuna_logger = logging.getLogger("optuna")
    previous_level = optuna_logger.level
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optuna_logger.setLevel(logging.ERROR)
            plot_result = plot_func(study)
            fig = _figure_from_plot_result(plot_result)
            if fig is None:
                raise RuntimeError("Matplotlib figure bulunamadi.")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        return str(save_path)
    except Exception as exc:
        plt.close("all")
        print(f"[UYARI] HPO {label} grafigi kaydedilemedi: {exc}")
        return None
    finally:
        optuna_logger.setLevel(previous_level)


def _save_hpo_visualizations(study: Any, study_dir: Path) -> dict[str, str]:
    """Optuna study icin analiz grafiklerini kaydet."""
    try:
        from optuna.visualization import matplotlib as optuna_mpl
    except Exception as exc:
        print(f"[UYARI] Optuna matplotlib gorselleri yuklenemedi: {exc}")
        return {}

    visuals_dir = study_dir / "gorseller"
    plotters = {
        "optimization_history": optuna_mpl.plot_optimization_history,
        "param_importances": optuna_mpl.plot_param_importances,
        "parallel_coordinate": optuna_mpl.plot_parallel_coordinate,
        "slice": optuna_mpl.plot_slice,
    }

    artifacts: dict[str, str] = {}
    for key, plot_func in plotters.items():
        saved_path = _save_single_hpo_plot(
            study,
            plot_func,
            visuals_dir / f"hpo_{key}.png",
            key,
        )
        if saved_path is not None:
            artifacts[key] = saved_path
    return artifacts


def _save_study_artifacts(
    study,
    args: argparse.Namespace,
    study_dir: Path,
    study_name: str,
    final_run: dict[str, Any] | None,
    existing_trial_count: int,
    trials_executed_this_run: int,
) -> None:
    trials_df = study.trials_dataframe()
    trials_df.to_csv(study_dir / "trial_history.csv", index=False)
    if getattr(args, "no_hpo_plots", False):
        hpo_visualizations: dict[str, str] = {}
    else:
        hpo_visualizations = _save_hpo_visualizations(study, study_dir)

    state_counts = Counter(str(trial.state) for trial in study.trials)
    best_trial = study.best_trial
    summary = {
        "study_name": study_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "search_type": "bayesian_tpe",
        "hpo_folds": args.hpo_folds,
        "metric": args.metric,
        "direction": _selection_mode_for_metric(args.metric),
        "trials_requested": args.trials,
        "timeout_seconds": args.timeout,
        "existing_trials_before_run": existing_trial_count,
        "trials_executed_this_run": trials_executed_this_run,
        "study_output_dir": str(study_dir),
        "storage": args.storage,
        "state_counts": dict(state_counts),
        "search_space": _search_space_summary(args),
        "best_trial": {
            "number": best_trial.number,
            "value": best_trial.value,
            "params": best_trial.params,
            "user_attrs": best_trial.user_attrs,
        },
        "final_run_dir": str(final_run["output_root"]) if final_run is not None else None,
        "final_report_path": str(final_run["report_path"]) if final_run is not None else None,
        "visualizations": hpo_visualizations,
    }
    _write_json(study_dir / "study_summary.json", summary)


def _prepare_storage_url(storage: str | None, study_dir: Path) -> str | None:
    """SQLite storage URL'sindeki dosya icin ust klasoru olusturup yolu normalize eder.

    sqlite3 eksik dizinleri kendisi yaratmaz; klasor yoksa
    'unable to open database file' hatasi verir. Burada hem mutlak/goreli
    yollari ayristirir, hem Windows ters egik cizgilerini duzeltir, hem de
    parent klasoru garanti altina aliriz.
    """
    if not storage:
        return storage
    if not storage.startswith("sqlite:"):
        return storage

    raw = storage[len("sqlite:"):]
    # SQLAlchemy formati: "sqlite:///relative.db" veya "sqlite:////abs/path.db"
    # Windows mutlak: "sqlite:///C:/path/file.db" (uc slash + surucu harfi)
    leading_slashes = len(raw) - len(raw.lstrip("/"))
    path_part = raw.lstrip("/")
    if not path_part:
        return storage  # bos / bellek-ici DB; dokunma

    # Windows'ta gelen ters slash'lari forward slash'a cevir
    path_part_norm = path_part.replace("\\", "/")
    db_path = Path(path_part_norm)
    if not db_path.is_absolute():
        # Goreli yollari study_dir altina cek; calisma dizinine bagimliligi azaltir
        db_path = (study_dir / db_path).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    abs_str = str(db_path).replace("\\", "/")
    # Mutlak yol icin SQLAlchemy 'sqlite:///' + 'C:/...' bekler (toplam 3 slash)
    if leading_slashes >= 3 or db_path.is_absolute():
        return f"sqlite:///{abs_str}"
    # Goreli senaryo (yukarida absolute'a cevirdik ama yine de guvenli olarak)
    return f"sqlite:///{abs_str}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_search_args(args)
    except (FileNotFoundError, ModuleNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[HATA] {exc}")
        return 1

    study_name = args.study_name or _default_study_name(args)
    study_dir = _resolve_study_dir(args, study_name)
    study_dir.mkdir(parents=True, exist_ok=True)
    try:
        args.storage = _prepare_storage_url(args.storage, study_dir)
    except OSError as exc:
        print(f"[HATA] Storage yolu hazirlanirken hata: {exc}")
        return 1

    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        n_startup_trials=args.n_startup_trials,
        multivariate=True,
        warn_independent_sampling=False, #Mevcut durumda rastgele degerlerin denenmesini istedigimiz icin bu uyari kapatildi.
    )
    if args.model == "xgboost":
        # XGBoost objective trial.report() cagirmiyor; MedianPruner anlamsiz
        # calisirdi, bu yuzden NopPruner ile devre disi birakilir.
        pruner = optuna.pruners.NopPruner()
        print("[INFO] XGBoost modunda epoch-bazli pruning desteklenmiyor; pruner devre disi.")
    elif args.hpo_folds > 1:
        # CV modunda her trial K bagimsiz egitimden olusur; epoch-bazli rapor
        # tek fold'a karsilik gelmedigi icin median pruning anlamli degil.
        pruner = optuna.pruners.NopPruner()
        print(
            "[INFO] HPO CV modunda (--hpo-folds>1) epoch-bazli pruning devre disi; "
            "her trial K fold'un val ortalamasiyla degerlendirilir."
        )
    else:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=args.pruner_startup_trials,
            n_warmup_steps=args.pruner_warmup_epochs,
        )
    study = optuna.create_study(
        study_name=study_name,
        direction=_selection_mode_for_metric(args.metric),
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=bool(args.storage),
    )
    existing_trial_count = len(study.trials)
    trials_to_run = _remaining_trials_to_run(args.trials, existing_trial_count)

    print(f"[INFO] Bayes search basliyor: study={study_name}, metric={args.metric}, trials={args.trials}")
    print(f"[INFO] Ciktilar: {study_dir}")
    if args.storage:
        print(
            f"[INFO] Storage study trial durumu: mevcut={existing_trial_count}, "
            f"hedef_toplam={args.trials}, bu_calismada={trials_to_run}"
        )

    if trials_to_run > 0:
        if args.model == "xgboost":
            objective_fn = hpo_xgb._xgb_objective_factory(args, study_dir)
        else:
            objective_fn = hpo_dl._objective_factory(args, study_dir)
        study.optimize(
            objective_fn,
            n_trials=trials_to_run,
            timeout=args.timeout,
            catch=(RuntimeError, ValueError, FileNotFoundError),
            gc_after_trial=True,
            n_jobs=int(getattr(args, "n_jobs", 1)),
        )
    else:
        print("[INFO] Mevcut study zaten istenen toplam trial sayisina ulasmis; yeni trial calistirilmadi.")

    completed_trials = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    if not completed_trials:
        print("[HATA] Hic tamamlanan trial yok. Arama sonlandirildi.")
        return 1

    final_run = None
    if not args.skip_final_train:
        print(
            "[INFO] En iyi trial bulundu: "
            f"#{study.best_trial.number} ({args.metric}={study.best_value:.4f}). "
            "Tum trainval uzerinde final egitim baslatiliyor."
        )
        if args.model == "xgboost":
            final_run = hpo_xgb._run_final_xgb_training(
                args=args,
                study_dir=study_dir,
                best_params=study.best_trial.params,
                study_name=study_name,
                best_trial_number=study.best_trial.number,
                best_iteration=study.best_trial.user_attrs.get("best_iteration"),
            )
        else:
            final_run = hpo_dl._run_final_training(
                args=args,
                study_dir=study_dir,
                best_params=study.best_trial.params,
                study_name=study_name,
                best_trial_number=study.best_trial.number,
                best_epoch=study.best_trial.user_attrs.get("best_epoch"),
            )

    _save_study_artifacts(
        study=study,
        args=args,
        study_dir=study_dir,
        study_name=study_name,
        final_run=final_run,
        existing_trial_count=existing_trial_count,
        trials_executed_this_run=trials_to_run,
    )

    print(
        f"[OK] Bayes search tamamlandi. En iyi trial=#{study.best_trial.number}, "
        f"{args.metric}={study.best_value:.4f}"
    )
    print(f"[OK] Ozet: {study_dir / 'study_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
