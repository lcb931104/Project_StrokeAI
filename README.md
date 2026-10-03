![](LOGO/LOGO.png)

---

StrokeAI 是一套以 Python 開發的腦部醫學影像分析工具，支援載入 NIfTI 與 DICOM 影像，並透過 nnU-Net 自動進行腦部區域分割與量化分析。
系統提供直覺化 GUI，可切換 Axial、Sagittal、Coronal 與 3V 三視角顯示模式，協助使用者檢視不同方向的腦部切面。使用者也可開啟 Overlay 疊圖，並調整透明度，以比對原始影像與分割結果。

---

主要功能包含：
- 支援 NIfTI（.nii、.nii.gz）及 DICOM 影像輸入
- 自動轉換 DICOM 影像為 NIfTI 格式
- 整合 nnU-Net 腦部影像分割模型
- 顯示原始影像、ROI 與 Atlas 分割結果
- 支援 Ax、Sag、Cor、3V 多視角檢視
- 可依 Slice 控制切面位置
- 支援分割結果 Overlay 與透明度調整
- 進行腦部區域體積與比例量化分析
- 匯出分析流程所需的中間結果與最終結果

StrokeAI 的目標是將醫學影像前處理、AI 分割、視覺化與量化分析整合在同一個操作介面中，降低影像分析流程的操作門檻，提升腦中風相關影像研究與判讀的效率。

---


