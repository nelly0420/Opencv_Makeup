// queue.js
class Queue {
    constructor(size) {
        this.size = size;
        this.key = 'queueData';  // 로컬 스토리지에서 사용할 키
        this.load();
    }

    load() {
        let storedQueue = localStorage.getItem(this.key);
        this.queue = storedQueue ? JSON.parse(storedQueue) : [];
    }

    save() {
        localStorage.setItem(this.key, JSON.stringify(this.queue));
    }

    enqueue(item) {
        if (this.queue.length >= this.size) {
            this.queue.shift();  // 큐의 크기를 초과하면 첫 번째 요소 제거
        }
        this.queue.push(item);  // 새로운 요소 추가
        this.save();
    }

    dequeue() {
        let item = this.queue.shift();  // 가장 오래된 요소 제거 및 반환
        this.save();
        return item;
    }

    getQueue() {
        return this.queue;
    }
}

// Make the Queue class available globally
window.Queue = Queue;
