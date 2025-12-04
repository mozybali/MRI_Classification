"""
plot_utils.py
-------------
Tüm grafik çizim fonksiyonları.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .config import (
    OUTPUT_DIR,
    RANDOM_SEED,
    N_IMAGES_INTENSITY_SAMPLE,
    N_IMAGES_FOR_EMBEDDING,
)
from .io_utils import (
    goruntu_yukle_yoksa_gri,
    normalize_goruntu,
    rastgele_piksel_ornekle,
)
from .stats_utils import gomuleme_icin_ozellik_matrisi


def _kaydet_ve_kapat(fig, dosya_adi: str):
    yol = os.path.join(OUTPUT_DIR, dosya_adi)
    fig.savefig(yol, dpi=200)
    plt.close(fig)
    print(f"[KAYDEDILDI] {yol}")


def sinif_dagilimi_ciz(df):
    fig = plt.figure(figsize=(6, 4))
    sns.countplot(x="label_name", data=df)
    plt.xlabel("Sınıf Etiketi")
    plt.ylabel("Adet")
    plt.title("Sınıf (Label) Dağılımı")
    plt.tight_layout()
    _kaydet_ve_kapat(fig, "sinif_dagilimi.png")


def boyut_dagilimlari_ciz(df):
    # Genişlik ve yükseklik histogramları
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df["genislik"], kde=False, ax=axes[0])
    axes[0].set_title("Genişlik Dağılımı")
    axes[0].set_xlabel("Genişlik (piksel)")
    axes[0].set_ylabel("Adet")

    sns.histplot(df["yukseklik"], kde=False, ax=axes[1])
    axes[1].set_title("Yükseklik Dağılımı")
    axes[1].set_xlabel("Yükseklik (piksel)")
    axes[1].set_ylabel("Adet")

    plt.tight_layout()
    _kaydet_ve_kapat(fig, "boyut_histogramlari.png")

    # Genişlik vs yükseklik scatter (renk: label)
    fig = plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df, x="genislik", y="yukseklik", hue="label_name", alpha=0.7)
    plt.title("Genişlik vs Yükseklik (sınıfa göre)")
    plt.xlabel("Genişlik")
    plt.ylabel("Yükseklik")
    plt.tight_layout()
    _kaydet_ve_kapat(fig, "genislik_yukseklik_scatter.png")


def yogunluk_ozellik_kutu_grafikleri_ciz(df):
    ozellikler = ["int_ort", "int_std", "int_p1", "int_p99"]
    fig = plt.figure(figsize=(12, 6))
    df_melt = df.melt(id_vars="label_name", value_vars=ozellikler,
                      var_name="ozellik", value_name="deger")
    sns.boxplot(data=df_melt, x="ozellik", y="deger", hue="label_name")
    plt.title("Yoğunluk Tabanlı Özniteliklerin Sınıfa Göre Dağılımı")
    plt.xticks(rotation=45)
    plt.tight_layout()
    _kaydet_ve_kapat(fig, "yogunluk_ozellik_kutu_grafikleri.png")


def global_yogunluk_histogramlari_ciz(df):
    """Her sınıf için birkaç görüntüden piksel örnekleyip KDE çiz."""
    etiketler = df["label_name"].unique()
    n_per_label = max(1, N_IMAGES_INTENSITY_SAMPLE // len(etiketler))

    fig = plt.figure(figsize=(10, 6))

    for label in etiketler:
        alt = df[df["label_name"] == label]
        alt = alt.sample(min(len(alt), n_per_label), random_state=RANDOM_SEED)

        ornekler = []
        for _, row in alt.iterrows():
            piksel = rastgele_piksel_ornekle(row["filepath"])
            if piksel.size > 0:
                ornekler.append(piksel)

        if len(ornekler) == 0:
            continue

        tum_piksel = np.concatenate(ornekler)
        sns.kdeplot(tum_piksel, label=f"Label {label}", linewidth=1)

    plt.title("Sınıflara Göre Global Yoğunluk Dağılımı (KDE)")
    plt.xlabel("Yoğunluk")
    plt.ylabel("Yoğunluk Tahmini")
    plt.legend()
    plt.tight_layout()
    _kaydet_ve_kapat(fig, "global_yogunluk_kde_sinif_bazli.png")


def rastgele_ornek_gorseller_ciz(df, n_per_label: int = 4):
    """Her sınıftan n_per_label adet rastgele JPEG görüntü göster."""
    etiketler = sorted(df["label_name"].unique())
    n_satir = len(etiketler)
    n_sutun = n_per_label

    fig, axes = plt.subplots(n_satir, n_sutun, figsize=(3 * n_sutun, 3 * n_satir))
    if n_satir == 1:
        axes = np.array([axes])

    for i, label in enumerate(etiketler):
        alt = df[df["label_name"] == label]
        n_ornek = min(len(alt), n_per_label)
        if n_ornek == 0:
            continue
        alt = alt.sample(n_ornek, random_state=RANDOM_SEED)

        for j, (_, row) in enumerate(alt.iterrows()):
            ax = axes[i, j]
            goruntu = goruntu_yukle_yoksa_gri(row["filepath"])
            if goruntu.ndim == 2:
                goruntu_norm = normalize_goruntu(goruntu)
                ax.imshow(goruntu_norm, cmap="gray")
            else:
                # RGB ise [0,1] aralığına getirelim
                goruntu_norm = goruntu / 255.0
                ax.imshow(goruntu_norm.astype("float32"))
            ax.axis("off")
            ax.set_title(f"Label {label}\nID: {row['id']}", fontsize=8)

        for j in range(n_ornek, n_per_label):
            axes[i, j].axis("off")

    plt.suptitle("Her Sınıftan Rastgele Örnek Görüntüler", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _kaydet_ve_kapat(fig, "rastgele_ornek_gorseller.png")


def pca_gomuleme_ciz(df):
    """Basit öznitelikler üzerinden 2B PCA gömme grafiği."""
    if len(df) < 2:
        print("[UYARI] PCA için yeterli örnek yok.")
        return

    if len(df) > N_IMAGES_FOR_EMBEDDING:
        df_emb = df.sample(N_IMAGES_FOR_EMBEDDING, random_state=RANDOM_SEED)
    else:
        df_emb = df.copy()

    X, ozellik_kolonlari = gomuleme_icin_ozellik_matrisi(df_emb)

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X)

    fig = plt.figure(figsize=(6, 5))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df_emb["label_name"], alpha=0.8)
    plt.title("PCA (2B) - Basit Öznitelikler")
    plt.xlabel("Bileşen 1")
    plt.ylabel("Bileşen 2")
    plt.tight_layout()
    _kaydet_ve_kapat(fig, "pca_gomuleme.png")


def tsne_gomuleme_ciz(df):
    """Aynı öznitelikler üzerinden 2B t-SNE gömme grafiği."""
    if len(df) < 3:
        print("[UYARI] t-SNE için yeterli örnek yok.")
        return

    if len(df) > N_IMAGES_FOR_EMBEDDING:
        df_emb = df.sample(N_IMAGES_FOR_EMBEDDING, random_state=RANDOM_SEED)
    else:
        df_emb = df.copy()

    X, ozellik_kolonlari = gomuleme_icin_ozellik_matrisi(df_emb)

    tsne = TSNE(
        n_components=2,
        perplexity=min(30, max(5, len(df_emb) // 2)),
        learning_rate="auto",
        init="random",
        random_state=RANDOM_SEED,
    )
    X_tsne = tsne.fit_transform(X)

    fig = plt.figure(figsize=(6, 5))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=df_emb["label_name"], alpha=0.8)
    plt.title("t-SNE (2B) - Basit Öznitelikler")
    plt.xlabel("Boyut 1")
    plt.ylabel("Boyut 2")
    plt.tight_layout()
    _kaydet_ve_kapat(fig, "tsne_gomuleme.png")
