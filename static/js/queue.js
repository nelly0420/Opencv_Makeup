class Queue {
    constructor(maxSize, storageKey = 'queue') {
        this.maxSize = maxSize;
        this.storageKey = storageKey;
        this.queue = this.loadQueue();
    }

    // 로컬 스토리지에서 큐를 로드
    loadQueue() {
        const storedQueue = JSON.parse(localStorage.getItem(this.storageKey));
        return storedQueue || []; // 저장된 큐가 없으면 빈 배열로 초기화
    }

    // 현재 큐를 로컬 스토리지에 저장
    saveQueue() {
        localStorage.setItem(this.storageKey, JSON.stringify(this.queue));
    }

    // 큐에 아이템 추가 (enqueue)
    enqueue(item) {
        this.queue.push(item);
        if (this.queue.length > this.maxSize) {
            this.queue.shift(); // 큐의 크기를 초과하면 첫 번째 아이템 제거
        }
        this.saveQueue(); // 변경된 큐를 로컬 스토리지에 저장
    }

    // 큐에서 아이템 제거 (dequeue) - 개별삭제
    dequeue() {
        const item = this.queue.shift();
        this.saveQueue(); // 변경된 큐를 로컬 스토리지에 저장
        return item;
    }

    // 큐 상태를 반환
    getQueue() {
        return this.queue;
    }

    // 큐 비우기 - 전체삭제
    clear() {
        this.queue = [];
        this.saveQueue(); // 로컬 스토리지에서도 큐 삭제
    }
}

// Make the Queue class available globally
window.Queue = Queue;
