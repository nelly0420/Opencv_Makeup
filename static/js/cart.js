// cart.js

// Simulate cart data (this could be replaced with actual server-side logic)
let cart = JSON.parse(localStorage.getItem('cart')) || {};

// Maximum allowed items in the cart
const MAX_CART_ITEMS = 3;

// Function to add an item to the cart
function addToCart(prdCode) {
    if (!prdCode) {
        console.error('Product code is missing');
        return;
    }

    let cart = JSON.parse(localStorage.getItem('cart')) || {};
    let currentCount = Object.keys(cart).length;

    // Check if the cart exceeds the maximum allowed items
    if (currentCount >= MAX_CART_ITEMS) {
        alert(`You can only add up to ${MAX_CART_ITEMS} items to the cart.`);
        return;
    }

    if (cart[prdCode]) {
        cart[prdCode].quantity += 1;
    } else {
        // Example product data
        cart[prdCode] = {
            PrdName: 'Example Product',
            color: 'Red',
            price: '100',
            quantity: 1,
        };
    }

    localStorage.setItem('cart', JSON.stringify(cart));
    localStorage.setItem('cartCount', currentCount + 1);
    $('#cookie-count').text(currentCount + 1);
}
