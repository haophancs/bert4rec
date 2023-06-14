// This file is placed in the "static" directory alongside style.css

// Code for handling clicks on the movie links and preventing the default behavior
document.addEventListener('click', (event) => {
  const link = event.target.closest('a');
  if (link) {
    event.preventDefault();
    const href = link.getAttribute('href');
    navigateTo(href);
  }
});

// Function to navigate to a new page using the browser's History API
function navigateTo(url) {
  history.pushState(null, null, url);
  handlePageLoad();
}

// Function to handle page load or navigation within the app
function handlePageLoad() {
  const currentPath = window.location.pathname;

  if (currentPath === '/') {
    fetchMovies();
  } else if (currentPath.startsWith('/movie/')) {
    const movieId = currentPath.split('/').pop();
    fetchMovie(movieId);
    fetchRecommendations(movieId);
  }
}

// Call handlePageLoad on page load and when the History API's popstate event is triggered (e.g., back/forward button)
window.addEventListener('DOMContentLoaded', handlePageLoad);
window.addEventListener('popstate', handlePageLoad);
