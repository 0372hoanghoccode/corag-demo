# Kế hoạch thuyết trình: CoRAG-inspired Iterative RAG

## Mục tiêu buổi trình bày

- Cho khán giả thấy rõ khác biệt giữa RAG truyền thống và CoRAG-inspired bằng demo chạy thật.
- Trình bày trung thực: đây là bản CoRAG-inspired, simplified để demo.
- Giúp người nghe biết khi nào nên ưu tiên tốc độ, khi nào nên ưu tiên độ phủ bằng chứng.

## Mở bài đơn giản (dùng nguyên văn)

"Hôm nay mình demo nhanh hai cách làm RAG trên cùng một câu hỏi: RAG truyền thống và CoRAG-inspired.
Điểm chính mình muốn mọi người nhìn là: cách truy xuất khác nhau sẽ ảnh hưởng trực tiếp tới chất lượng bằng chứng của câu trả lời."

## Khung tổng thể

- Thời lượng: 30-45 phút.
- Đối tượng: IT / Software Engineering.
- Ngôn ngữ: Tiếng Việt.
- Cấu trúc: 5 phần + Q&A.
- Luồng demo chính: 3-hop rồi 5-hop.

## Kịch bản chi tiết theo thời gian

### 1) Hook (0-5 phút)

- Mục tiêu phần này:
  - Tạo cùng kỳ vọng: câu hỏi nhiều bước thì cần bằng chứng nhiều bước.
- Cách nói ngắn:
  - "Cùng một câu hỏi, hai hệ thống có thể đều trả lời trôi chảy. Nhưng cái mình quan tâm là bằng chứng có đủ chưa."
- Câu chuyển:
  - "Giờ mình đi từ nền tảng RAG trước, rồi vào CoRAG-inspired để thấy sự khác nhau ở cách truy xuất."

### Hướng tìm câu trả lời cho phần Hook

- Khi bị hỏi "vì sao phải so bằng chứng":
  - Trả lời: vì câu trả lời nghe hợp lý có thể vẫn thiếu nguồn hỗ trợ.
- Khi bị hỏi "mục tiêu demo là gì":
  - Trả lời: so sánh hành vi truy xuất, không phải thi xem model nào văn hay hơn.

### 2) Nền tảng RAG truyền thống (5-12 phút)

- Ý chính:
  - RAG truyền thống: truy xuất một lần, trả lời một lần.
  - Ưu điểm: nhanh, gọn, chi phí thấp.
  - Giới hạn: câu hỏi nhiều vế dễ thiếu chứng cứ nếu lần truy xuất đầu chưa đủ.
- Ví dụ đời thường:
  - "Giống như tìm tài liệu một lượt rồi chốt luôn kết luận."
- Câu chuyển:
  - "Nếu hệ thống được quyền kiểm tra phần thiếu rồi truy xuất bổ sung thì sao?"

### Hướng tìm câu trả lời cho phần RAG

- Nếu bị hỏi "RAG có tệ không":
  - Trả lời: không tệ, rất tốt cho câu hỏi đơn giản và cần phản hồi nhanh.
- Nếu bị hỏi "lỗi chính của RAG là gì":
  - Trả lời: không tự quay lại lấy thêm bằng chứng khi ngữ cảnh ban đầu thiếu.

### 3) CoRAG-inspired hoạt động thế nào (12-22 phút)

- Ý chính:
  1. Tách câu hỏi thành các phần cần chứng cứ.
  2. Đánh dấu phần nào đã đủ, phần nào thiếu.
  3. Sinh truy vấn phụ, truy xuất thêm theo vòng lặp đến khi đủ hoặc chạm max_steps.
- Cách nói trung thực:
  - "Đây là bản demo theo tinh thần CoRAG, không phải bản tái hiện đầy đủ paper."
- Trade-off cần nhấn mạnh:
  - Chậm hơn do nhiều bước.
  - Tốn token hơn do có thêm bước đánh giá và sinh truy vấn phụ.
  - Đổi lại thường đầy bằng chứng hơn ở câu khó.
  - Giảm rủi ro hallucination vì mô hình bị buộc bám vào bằng chứng đã truy xuất.

### Hướng tìm câu trả lời cho phần CoRAG

- Nếu bị hỏi "khác paper ở đâu":
  - Trả lời: đây là phiên bản đơn giản hóa để demo nguyên lý vòng lặp truy xuất.
- Nếu bị hỏi "vì sao chậm":
  - Trả lời: có thêm các vòng đánh giá thiếu đủ và truy xuất bổ sung.
- Nếu bị hỏi "chậm có đáng không":
  - Trả lời: đáng khi bài toán cần độ tin cậy bằng chứng cao.
- Nếu bị hỏi "CoRAG giúp gì với hallucination":
  - Trả lời: CoRAG giúp giảm ảo giác bằng cách ép hệ thống quay lại tìm phần còn thiếu trước khi kết luận.

### 4) Demo live (22-35 phút)

#### Câu 1: 3-hop

- Mục tiêu:
  - Baseline khách quan, cho thấy có trường hợp hai hệ thống gần nhau.
- Cách dẫn:
  - "Mình chạy 3-hop trước để mọi người có mốc so sánh công bằng."
- Điểm cần chỉ:
  - steps, total docs, latency, chất lượng bằng chứng.

#### Câu 2: 5-hop

- Mục tiêu:
  - Thể hiện vùng mà CoRAG-inspired tạo khác biệt rõ.
- Cách dẫn:
  - "Giờ tăng độ khó lên 5-hop để xem khi thiếu bằng chứng thì hệ thống xử lý ra sao."
- Điểm cần chỉ:
  - CoRAG đi nhiều bước hơn.
  - Tập chứng cứ đầy hơn.
  - Độ trễ cao hơn là chi phí phải trả.

#### Lưu ý khi chạy live

- Không tua nhanh status, để khán giả thấy từng bước.
- Nếu bước lâu, nói rõ: "hệ thống đang bổ sung phần còn thiếu".
- Nếu 5-hop chạy lâu 40-50 giây, dùng câu này:
  - "Mọi người thấy không, hệ thống vừa phát hiện thiếu một mảnh thông tin nên đang tự truy xuất thêm để không trả lời thiếu bằng chứng."
- Chốt ngay sau mỗi câu:
  - "Nhìn vào bằng chứng trước, rồi mới nhìn tốc độ."

### Hướng tìm câu trả lời cho phần Demo

- Nếu bị hỏi "sao cùng model mà kết quả khác":
  - Trả lời: khác nhau chủ yếu ở chiến lược truy xuất, không chỉ ở model sinh câu trả lời.
- Nếu bị hỏi "đo chất lượng kiểu gì":
  - Trả lời: nhìn mức độ bao phủ của bằng chứng và độ nhất quán giữa các nguồn.
- Nếu bị hỏi "có thể fail không":
  - Trả lời: có thể, vì dữ liệu và retrieval luôn có nhiễu; mục tiêu là giảm rủi ro thiếu chứng cứ.
- Nếu bị hỏi "chi phí có tăng không":
  - Trả lời: có, vì CoRAG gọi LLM nhiều lần hơn; đây là đánh đổi giữa chi phí và độ tin cậy.

### 5) Kết luận và ứng dụng (35-40 phút)

- Kết theo 3 trục:
  - Chất lượng câu trả lời.
  - Độ trễ.
  - Chi phí suy luận.
- Câu kết chính:
  - "CoRAG không phải silver bullet, mà là công cụ đúng cho đúng bài toán."
- Mở rộng:
  - Agentic RAG cho các luồng cần phối hợp nhiều bước và nhiều công cụ.

### Hướng tìm câu trả lời cho phần Kết luận

- Nếu bị hỏi "vậy chọn cái nào":
  - Trả lời: câu dễ, cần nhanh thì RAG; câu nhiều bước, cần chắc bằng chứng thì CoRAG-inspired.
- Nếu bị hỏi "triển khai thực tế bắt đầu từ đâu":
  - Trả lời: bắt đầu từ use-case nhiều hop, bật loop có giới hạn max_steps, theo dõi latency và chất lượng.

### 6) Q&A (40-45 phút)

Chuẩn bị sẵn 6 câu trả lời ngắn:

1. Khi nào chỉ cần RAG truyền thống?
2. Khi nào bật CoRAG-inspired?
3. Vì sao CoRAG chậm hơn?
4. Demo này khác gì paper đầy đủ?
5. Đánh giá chất lượng khách quan bằng gì?
6. Nếu budget hạn chế thì tối ưu ra sao?
7. CoRAG có giảm hallucination không?
8. Vì sao tốn token hơn và có đáng không?

## Checklist trước giờ lên sân khấu

1. Rehearsal đủ 2 câu chính: 3-hop và 5-hop.
2. Xác nhận status theo bước hiển thị ổn định.
3. Chốt tham số demo để giảm dao động.
4. Giữ wording nhất quán:
   - CoRAG-inspired.
   - Simplified để demo.
   - Không claim full paper parity.
5. Chuẩn bị dự phòng:
   - Ảnh chụp kết quả.
   - Log terminal của các lần chạy.

## Script nói mẫu (dùng trực tiếp)

- "Mình không cố chứng minh CoRAG luôn thắng. Mình muốn chỉ ra khi nào vòng lặp truy xuất giúp câu trả lời chắc bằng chứng hơn."
- "Ở 3-hop có thể hai bên khá gần nhau. Ở 5-hop, khác biệt thường rõ hơn vì nhu cầu tổng hợp bằng chứng tăng mạnh."
- "Nếu ưu tiên tốc độ và chi phí cho câu đơn giản, RAG truyền thống vẫn rất hợp lý."
- "Nếu ưu tiên độ phủ bằng chứng cho câu nhiều bước, CoRAG-inspired phù hợp hơn."

## Plan B khi có sự cố

1. Nếu API chậm, chạy thẳng câu 5-hop để giữ thông điệp chính.
2. Nếu timeout, chuyển sang ảnh kết quả và log đã chuẩn bị.
3. Nếu thiếu thời gian, giảm phần nền tảng, giữ phần demo và kết luận.
