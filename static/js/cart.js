const MAX_ITEMS = 3;
let cart = [];

// 쿠키에서 장바구니 개수 가져오기
function getCookie(name) {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) return parts.pop().split(';').shift();
    return null; // 쿠키가 없을 때 null 반환
}

// 쿠키에 장바구니 개수 저장하기
function setCookie(name, value, days) {
    const d = new Date();
    d.setTime(d.getTime() + (days * 24 * 60 * 60 * 1000));
    const expires = `expires=${d.toUTCString()}`;
    document.cookie = `${name}=${value}; ${expires}; path=/`;
}

// 장바구니에 상품 추가
function addToCart(item) {
   
    
    if (cart.length >= MAX_ITEMS) {
        cart.shift(); // FIFO: 가장 오래된 상품 제거
    }
    cart.push(item);
    updateCartDisplay();
    setCookie('cart', JSON.stringify(cart), 1); // 쿠키에 장바구니 저장
}

// 장바구니 표시
function updateCartDisplay() {
    const cartContainer = document.getElementById('cartContainer');
    if (cartContainer) {
        cartContainer.innerHTML = '';
        if (cart.length > 0) {
            cart.forEach(item => {
                const div = document.createElement('div');
                div.className = 'cart-item';
                div.textContent = item || 'Unknown item'; // item이 null이나 undefined일 경우 처리
                cartContainer.appendChild(div);
            });
        } else {
            cartContainer.textContent = '장바구니가 비어 있습니다.';
        }
    } else {
        console.error('Cart container element not found.');
    }
}


// 장바구니 보기
function viewCart() {
    const cartCookie = getCookie('cart');
    if (cartCookie) {
        try {
            const parsedCart = JSON.parse(cartCookie);
            if (Array.isArray(parsedCart)) {
                cart = parsedCart;
                updateCartDisplay();
            } else {
                console.error("Invalid cart data in cookie");
                cart = []; // 오류 발생 시 빈 배열로 초기화
                updateCartDisplay();
            }
        } catch (e) {
            console.error("Error parsing cart data from cookie:", e);
            cart = []; // 오류 발생 시 빈 배열로 초기화
            updateCartDisplay();
        }
    } else {
        alert('장바구니에 항목이 없습니다.');
        cart = []; // 장바구니가 비어 있을 때 초기화
        updateCartDisplay();
    }
}


// 페이지 로드 시 장바구니 초기화
document.addEventListener('DOMContentLoaded', () => {
    viewCart();
});
