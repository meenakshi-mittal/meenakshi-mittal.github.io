const imageNames = [
    { name: 'self_portrait', label: '[R: (176, 37), G: (78, 29)]' },
    { name: 'sculpture', label: '[R: (140, -26), G: (33, -11)]' },
    { name: 'lady', label: '[R: (120, 13), G: (56, 9)]' },
    { name: 'harvesters', label: '[R: (124, 14), G: (60, 17)]' },
    { name: 'church', label: '[R: (58, -4), G: (25, 4)]' },
    { name: 'icon', label: '[R: (90, 23), G: (42, 17)]' },
    { name: 'onion_church', label: '[R: (107, 36), G: (51, 26)]' },
    { name: 'melons', label: '[R: (177, 13), G: (80, 10)]' },
    { name: 'monastery', label: '[R: (3, 2), G: (-3, 2)]' },
    { name: 'emir', label: '[R: (107, 40), G: (49, 24)]' },
    { name: 'cathedral', label: '[R: (12, 3), G: (5, 2)]' },
    { name: 'train', label: '[R: (85, 29), G: (41, 2)]' },
    { name: 'tobolsk', label: '[R: (7, 3), G: (3, 3)]' },
    { name: 'three_generations', label: '[R: (111, 9), G: (54, 12)]' }
];

const specialImageNames = [
    { name: 'cliff_house', label: '[R: (76, -8), G: (-3, -2)]' },
    { name: 'canoe', label: '[R: (134, -14), G: (14, -9)]' },
    { name: 'arch', label: '[R: (154, 34), G: (73, 23)]' }
];

const imageVariations = ['0_1_contrast', 'histo_eq', 'adapt_histo_eq', 'gray_world', 'avg_world'];

const labels = ['0-1 Contrast', 'Hist. Eq.', 'Adaptive Hist. Eq.', 'Gray Balance', 'Avg. Balance'];

function createImageBlock(imageId, imageName, label) {
    const imageContainer = document.createElement('div');

    // Create label for the image with the custom label string
    const labelElement = document.createElement('p');
    labelElement.className = 'image-label';
    labelElement.innerHTML = `${imageName}: ${label}`; // Use the label string here

    const imageElement = document.createElement('img');
    imageElement.src = `media/out_${imageVariations[0]}/${imageName}.jpg`;
    imageElement.alt = `Grid Image ${imageId}`;
    imageElement.id = `image-${imageId}`;

    const buttonGroup = document.createElement('div');
    buttonGroup.className = 'button-group';

    labels.forEach((label, index) => {
        const button = document.createElement('button');
        button.className = 'toggle-button';
        button.textContent = label;
        button.setAttribute('onmouseover', `toggleImage(${imageId}, ${index + 1}, this)`);
        buttonGroup.appendChild(button);
    });

    imageContainer.appendChild(labelElement);
    imageContainer.appendChild(imageElement);
    imageContainer.appendChild(buttonGroup);

    return imageContainer;
}

function populateImageGrid() {
    const imageGrid = document.getElementById('image-grid');
    imageNames.forEach((image, index) => {
        const imageBlock = createImageBlock(index + 1, image.name, image.label);
        imageGrid.appendChild(imageBlock);
    });
}

function populateSpecialImageGrid() {
    const specialImageGrid = document.getElementById('special-image-grid');
    specialImageNames.forEach((image, index) => {
        const imageBlock = createImageBlock(imageNames.length + index + 1, image.name, image.label);
        specialImageGrid.appendChild(imageBlock);
    });
}

// Reset active buttons on all images
function resetActiveButtons(imageId) {
    const buttonGroup = document.querySelector(`#image-${imageId}`).nextElementSibling;
    buttonGroup.querySelectorAll('.toggle-button').forEach(button => {
        button.classList.remove('active');
    });
}

function toggleImage(imageId, variation, button) {
    const imgElement = document.getElementById(`image-${imageId}`);
    const allImageNames = [...imageNames, ...specialImageNames];
    if (imgElement) {
        const imageName = allImageNames[imageId - 1].name;
        imgElement.src = `media/out_${imageVariations[variation - 1]}/${imageName}.jpg`;
        resetActiveButtons(imageId);
        button.classList.add('active');
    } else {
        console.log(`Image not found for ID: ${imageId}`);
    }
}

// Populate the grids on page load
window.onload = function () {
    populateImageGrid();
    populateSpecialImageGrid();
};
