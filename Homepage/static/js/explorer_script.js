const animationGallery = document.getElementById('animation-gallery');
const liveactionGallery = document.getElementById('liveaction-gallery');
const stockGallery = document.getElementById('stock-gallery');
const shortsGallery = document.getElementById('shorts-gallery');
const youtubeGallery = document.getElementById('youtube-gallery');

const videoPlayer = document.getElementById('videoPlayer');
const videoDescriptions = document.getElementById('videoDescriptions');
const versionCheckboxes = document.querySelectorAll('input[type="checkbox"]');
const loading = document.getElementById('loading');

// Show loading animation
loading.style.display = 'block';

// Load video URLs and thumbnail URLs from a local file
fetch('static/data/video_data.json')
// fetch('https://raw.githubusercontent.com/tarsier-vlm/tarsier-vlm.github.io/main/static/data/video_data.json')
    .then(response => response.json())
    .then(data => {
        categorized_data = [data['Animation'], data['Live-action'], data['Stock'], data['Shorts'], data['YouTube']]
        gallery_list = [animationGallery, liveactionGallery, stockGallery, shortsGallery, youtubeGallery]

        for (let i = 0; i < 5; i++) {
            category_data = categorized_data[i]
            category_gallery = gallery_list[i]
            // Generate movie thumbnails
            category_data.forEach((video, index) => {
                const col = document.createElement('div');
                col.className = 'col-md-2 col-sm-4 mb-4';
    
                const thumbnail = document.createElement('img');
                thumbnail.src = `static/thumbnails/${video.idx}.jpg`;
                thumbnail.alt = `Video ${index}`;
                thumbnail.className = 'img-fluid video-thumbnail';
                thumbnail.setAttribute('data-idx', video.idx);
                thumbnail.setAttribute('data-url', video.video_url);
                thumbnail.setAttribute('data-descriptions', JSON.stringify(video.descriptions));
    
                col.appendChild(thumbnail);
                category_gallery.appendChild(col);
            });
        }

        // Play video when thumbnail is clicked
        const videoThumbnails = document.querySelectorAll('.video-thumbnail');
        videoThumbnails.forEach(thumbnail => {
            thumbnail.addEventListener('click', () => {
                const videoIdx = thumbnail.getAttribute('data-idx');
                const descriptions = JSON.parse(thumbnail.getAttribute('data-descriptions'));
                // videoPlayer.querySelector('source').src = `videos/${videoIdx}.mp4`;
                videoPlayer.querySelector('source').src = thumbnail.getAttribute('data-url');
                videoPlayer.load();

                // Clear previous descriptions
                videoDescriptions.innerHTML = '';
                
                // Get selected description versions
                var selectedVersions = Array.from(versionCheckboxes)
                    .filter(checkbox => checkbox.checked)
                    .map(checkbox => checkbox.value);
                selectedVersions = ["Human", ...selectedVersions]
                selectedVersions.forEach(version => {
                    const p = document.createElement('p');
                    const versionTag = document.createElement('strong');
                    versionTag.textContent = `${version}: `;
                    p.appendChild(versionTag);
                    p.appendChild(document.createTextNode(descriptions[version]));
                    videoDescriptions.appendChild(p);
                }) 
                // Update the video player title with the video index
                const videoModalLabel = document.getElementById('videoModalLabel');
                videoModalLabel.textContent = `Video ${videoIdx}`;
                $('#videoModal').modal('show');
            });
        });

        // Hide loading animation
        loading.style.display = 'none';
    }
)