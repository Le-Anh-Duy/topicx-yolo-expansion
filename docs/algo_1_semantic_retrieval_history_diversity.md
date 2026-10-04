# Algo 1 — Semantic Retrieval + History-aware Visual Diversity

Ngày ghi nhận: 2026-10-04.

Trạng thái: baseline đã thống nhất ở mức thiết kế; chưa triển khai, chưa có kết quả thực nghiệm.

## 1. Mục đích

Với một topic/requirement, đề xuất nhiều vòng batch ảnh để annotate và bổ sung vào dữ liệu huấn luyện. Mỗi batch phải phù hợp về ngữ nghĩa và hạn chế tương đồng với cả các ảnh trong batch hiện tại lẫn dữ liệu đã thu thập cho requirement ở những vòng trước.

Thiết lập khởi đầu đang thảo luận:

- Retrieval: top-K, với K = 500 ảnh.
- Batch: B = 16 ảnh mỗi vòng.
- Filter: loại ảnh có cosine similarity với topic nhỏ hơn threshold τ.
- Selection: greedy farthest-first có xét lịch sử.

K và B là cấu hình khởi đầu; τ và encoder cụ thể chưa chốt.

## 2. Phạm vi baseline

Baseline dùng text/image embeddings và lịch sử acquisition. Sau mỗi vòng có thể train lại YOLO, nhưng prediction, uncertainty, loss và AP của YOLO chưa tham gia điểm selection.

Do đó, thuật toán tìm ảnh bổ sung cho dữ liệu đã có; chưa trực tiếp tìm ảnh khắc phục điểm yếu của detector. Việc giữ năng lực base classes thuộc training/replay recipe và được đánh giá riêng.

## 3. Đầu vào và ký hiệu

| Ký hiệu | Ý nghĩa |
|---|---|
| q | Topic/requirement được giữ cố định trong thí nghiệm |
| U_t | Pool ảnh còn có thể chọn ở vòng t, không gồm dev/test |
| K | Số ảnh retrieval tối đa trước filter |
| B | Số ảnh mong muốn trong batch |
| τ | Ngưỡng cosine với topic |
| e_q | Text embedding đã chuẩn hóa |
| e_x | Image embedding dùng cho semantic relevance, đã chuẩn hóa |
| v_x | Visual embedding dùng cho diversity, đã chuẩn hóa |
| H_t | Ảnh đã được chấp nhận và annotate cho requirement này trước vòng t |
| C_t | Candidate shortlist sau retrieval và filter |
| S | Batch đang xây |

Đề xuất khởi đầu: dùng cùng image embedding cho retrieval và diversity, tức v_x = e_x. Encoder được giữ cố định qua các vòng để khoảng cách có cùng ý nghĩa. Dùng encoder khác hoặc region embedding là biến thể nghiên cứu sau.

H_t ghi nhận toàn bộ ảnh đã annotate cho requirement, gồm cả ảnh không chứa novel class nếu có. Đây là lịch sử acquisition, không phải tập ảnh đã được chứng minh hữu ích hay đã được YOLO học tốt.

## 4. Hai đại lượng giữ nguyên qua các vòng

Semantic relevance:

\[
r(x,q)=\cos(e_x,e_q).
\]

Visual distance:

\[
d(x,y)=\frac{1-\cos(v_x,v_y)}{2}.
\]

Relevance dùng để xếp hạng và lọc; distance dùng để chọn batch. Cosine với topic là score, không phải xác suất có class. Distance trong embedding là proxy cho khác biệt hình ảnh, không phải training utility.

Candidate set:

\[
C_t=\{x\in\operatorname{TopK}_{x\in U_t}r(x,q):r(x,q)\ge\tau\}.
\]

Các điều kiện cứng của requirement chỉ được lọc thêm khi có metadata đáng tin cậy. Một text embedding chưa bảo đảm đáp ứng đúng mọi điều kiện của requirement phức tạp.

## 5. Hàm mục tiêu cho batch

Khi chưa có lịch sử, muốn chọn batch có khoảng cách nhỏ nhất giữa các cặp càng lớn càng tốt:

\[
\max_{S\subseteq C_t,\ |S|=B}
\min_{\substack{x,y\in S\\x\ne y}}d(x,y).
\]

Khi có lịch sử, thêm điều kiện batch mới cũng cách xa dữ liệu đã thu thập:

\[
\max_{S\subseteq C_t,\ |S|=B}J_t(S),
\]

\[
J_t(S)=\min\left\{
\min_{\substack{x,y\in S\\x\ne y}}d(x,y),
\min_{\substack{x\in S\\h\in H_t}}d(x,h)
\right\}.
\]

Nếu H_t rỗng, bỏ thành phần khoảng cách tới lịch sử. Chỉ xét các cặp có ít nhất một ảnh thuộc batch mới; không đưa khoảng cách giữa hai ảnh lịch sử vào objective, vì batch mới không thể thay đổi chúng.

Relevance là điều kiện qua C_t, không cộng thêm một trọng số relevance vào objective của baseline này.

## 6. Cách giải gần đúng bằng greedy

Điểm thêm một ảnh:

\[
\delta_t(x,S)=\min_{y\in H_t\cup S}d(x,y).
\]

Quy trình:

1. Lấy top-K từ pool còn lại và filter theo τ để tạo C_t.
2. Khởi tạo S rỗng.
3. Nếu H_t rỗng, chọn ảnh có relevance cao nhất làm ảnh đầu tiên, vì chưa có ảnh để tính khoảng cách.
4. Nếu H_t không rỗng, chọn ảnh đầu tiên xa lịch sử nhất theo δ_t(x,∅).
5. Mỗi bước tiếp theo, chọn ảnh trong C_t chưa thuộc S có δ_t(x,S) lớn nhất.
6. Khi bằng điểm distance, ưu tiên relevance cao hơn; nếu vẫn bằng, dùng image ID để kết quả tái lập được.
7. Dừng khi đủ B ảnh.

Đây là quy tắc greedy tạo nghiệm gần đúng theo hướng tăng separation. Không tuyên bố đạt tối ưu toàn cục hay một approximation ratio cụ thể cho objective và distance đang dùng.

Với shortlist 500 ảnh và batch 16 ảnh, có thể lưu khoảng cách gần nhất của từng candidate. Khi thêm một ảnh, chỉ cập nhật khoảng cách tới ảnh mới. Phần xây batch cần khoảng O(KB) phép tính distance; khởi tạo khoảng cách tới lịch sử bằng cách trực tiếp cần thêm O(K|H_t|). Chi phí encode/retrieval tính riêng.

## 7. Quy trình nhiều vòng

**Retrieve → filter → greedy propose → chấp nhận/annotate → train/evaluate → cập nhật lịch sử → vòng tiếp theo.**

- Batch đang chờ xử lý được đánh dấu reserved để không xuất hiện đồng thời trong proposal khác.
- Sau khi batch A_t được chấp nhận và annotate, cập nhật H_{t+1} = H_t ∪ A_t và loại A_t khỏi pool có thể chọn.
- Nếu chỉ chấp nhận một phần batch, chỉ phần đó được cập nhật vào lịch sử.
- Batch bị từ chối không tự động trở thành dữ liệu lịch sử. Việc trả về pool hay block theo feedback là quy tắc sản phẩm cần xác định khi triển khai.
- Nếu retrieval lại trên pool còn lại mà encoder/topic không đổi, ranking nền vẫn có thể giữ nguyên; loại các ảnh đã dùng làm các ứng viên mới đi vào top-K.

Train/evaluate được thực hiện trong vòng lặp nghiên cứu, nhưng kết quả train không thay đổi selection score của Algo 1.

Nếu filter để lại dưới B ảnh, trả về trạng thái thiếu ứng viên. Có thể mở rộng K theo một quy tắc được công bố trước; không tự hạ τ hoặc lấy ảnh dưới threshold. Dừng khi hết ngân sách, hết ứng viên phù hợp hoặc theo số vòng đã định trước.

## 8. Đầu ra cần ghi lại

Mỗi proposal lưu topic, round ID, batch IDs, K, B, τ, encoder/version, history IDs, relevance scores và greedy selection order. Có thể kèm tóm tắt khoảng cách trong batch và tới lịch sử để giải thích vì sao batch được chọn.

Chốt batch IDs trước khi mở nhãn. Sau annotation mới báo positive images và instance yield thực tế.

## 9. Đối chứng để đánh giá

| Nhánh | Câu hỏi |
|---|---|
| Random toàn pool | Retrieval/selection có tốt hơn lấy ngẫu nhiên không? |
| Top-B relevance trong tập đủ threshold | Thêm diversity có ích không? |
| Greedy diversity chỉ trong batch | Diversity nội batch có ích không? |
| Algo 1: greedy diversity với H_t ∪ S | Xét lịch sử có thêm ích lợi qua nhiều vòng không? |

So sánh cùng ngân sách annotation tích lũy, cùng pool ban đầu, topic, encoder và training/replay recipe. Mỗi nhánh sau đó có lịch sử riêng do policy tạo ra. Nếu muốn so riêng một vòng, dùng chung history/checkpoint đầu vòng cho các nhánh.

Báo relevance, positive/instance yield, redundancy và AP novel/AP base sau mỗi vòng. Distance tăng chỉ cho thấy tối ưu proxy tốt hơn; cần downstream AP để kiểm tra chất lượng dữ liệu cho training.

Giả thuyết baseline: xét lịch sử giúp giảm việc thu thêm ảnh tương tự và cải thiện học novel class ở cùng ngân sách. Có thể bác bỏ nếu redundancy giảm nhưng AP không cải thiện, positive yield giảm quá nhiều, hoặc AP base suy giảm ngoài giới hạn đã chốt.

## 10. Giới hạn và câu hỏi để nghiên cứu tiếp

- Distance toàn ảnh có thể phản ánh nền, thời tiết nhiều hơn object cần học.
- Farthest-first có thể ưu tiên outlier dù vẫn vượt threshold semantic.
- Ảnh giống dữ liệu cũ vẫn có thể hữu ích nếu YOLO chưa học tốt; baseline chưa quan sát điều đó.
- Khi lịch sử lớn, một ảnh mới có thể gần ít nhất một ảnh cũ dù có khác biệt hữu ích cho detection.
- Top-K và threshold có thể loại positives khó trước khi diversity được áp dụng.
- Ngưỡng cosine phụ thuộc encoder và topic, cần được chốt/tune theo protocol dev; không mặc định một ngưỡng dùng tốt cho mọi topic.

Các hướng tiếp theo để nghiên cứu riêng: object/region diversity; cân bằng relevance–diversity; feedback từ lỗi detector sau seed; cách dùng lịch sử có chọn lọc. Các hướng này chưa thuộc Algo 1 đã chốt.

## 11. Quyết định hiện tại

Giữ metric semantic và visual distance qua các vòng; mở rộng objective để xét lịch sử; dùng greedy làm cách giải gần đúng. Đây là baseline nhiều vòng đầu tiên để đối chiếu các thuật toán tiếp theo.
