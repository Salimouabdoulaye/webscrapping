import scrapy
import re
import json
import random
import time
from urllib.parse import urljoin


class AmazonReviewsSpider(scrapy.Spider):
    name = 'amazon_reviews'
    allowed_domains = ['amazon.fr']  # Amazon France uniquement
    
    def __init__(self):
        self.products_scraped = 0
        self.target_products = 1000
        self.processed_asins = set()
        
        # STRATÉGIE MIXTE : Bestsellers + Produits mal notés + Promotions
        self.start_urls = self.generate_mixed_urls()
    
    def generate_mixed_urls(self):
        """Génère des URLs pour parcourir aléatoirement toutes les catégories de produits"""
        
        # CATÉGORIES PRINCIPALES avec recherches aléatoires
        category_searches = [
            # Electronics & High-Tech
            'https://www.amazon.fr/s?k=electronique&i=electronics',
            'https://www.amazon.fr/s?k=informatique&i=computers',
            'https://www.amazon.fr/s?k=smartphone&i=electronics',
            'https://www.amazon.fr/s?k=tablette&i=electronics',
            'https://www.amazon.fr/s?k=casque&i=electronics',
            'https://www.amazon.fr/s?k=ecouteurs&i=electronics',
            'https://www.amazon.fr/s?k=enceinte&i=electronics',
            'https://www.amazon.fr/s?k=montre+connectee&i=electronics',
            'https://www.amazon.fr/s?k=appareil+photo&i=electronics',
            'https://www.amazon.fr/s?k=gaming&i=electronics',
            
            # Maison & Jardin
            'https://www.amazon.fr/s?k=cuisine&i=kitchen',
            'https://www.amazon.fr/s?k=jardin&i=garden',
            'https://www.amazon.fr/s?k=electromenager&i=kitchen',
            'https://www.amazon.fr/s?k=decoration&i=home-garden',
            'https://www.amazon.fr/s?k=meuble&i=home-garden',
            'https://www.amazon.fr/s?k=bricolage&i=diy',
            'https://www.amazon.fr/s?k=outils&i=diy',
            'https://www.amazon.fr/s?k=maison&i=home-garden',
            'https://www.amazon.fr/s?k=rangement&i=home-garden',
            'https://www.amazon.fr/s?k=luminaire&i=lighting',
            
            # Mode & Beauté
            'https://www.amazon.fr/s?k=vetement&i=fashion',
            'https://www.amazon.fr/s?k=chaussures&i=shoes',
            'https://www.amazon.fr/s?k=accessoire&i=fashion',
            'https://www.amazon.fr/s?k=beaute&i=beauty',
            'https://www.amazon.fr/s?k=parfum&i=beauty',
            'https://www.amazon.fr/s?k=cosmetique&i=beauty',
            'https://www.amazon.fr/s?k=soin&i=beauty',
            'https://www.amazon.fr/s?k=bijoux&i=jewelry',
            'https://www.amazon.fr/s?k=montre&i=watches',
            'https://www.amazon.fr/s?k=sac&i=luggage',
            
            # Sport & Loisirs
            'https://www.amazon.fr/s?k=sport&i=sports',
            'https://www.amazon.fr/s?k=fitness&i=sports',
            'https://www.amazon.fr/s?k=musculation&i=sports',
            'https://www.amazon.fr/s?k=yoga&i=sports',
            'https://www.amazon.fr/s?k=running&i=sports',
            'https://www.amazon.fr/s?k=cyclisme&i=sports',
            'https://www.amazon.fr/s?k=natation&i=sports',
            'https://www.amazon.fr/s?k=camping&i=sports',
            'https://www.amazon.fr/s?k=randonnee&i=sports',
            'https://www.amazon.fr/s?k=peche&i=sports',
            
            # Jouets & Jeux
            'https://www.amazon.fr/s?k=jouet&i=toys',
            'https://www.amazon.fr/s?k=jeux&i=toys',
            'https://www.amazon.fr/s?k=puzzle&i=toys',
            'https://www.amazon.fr/s?k=lego&i=toys',
            'https://www.amazon.fr/s?k=peluche&i=toys',
            'https://www.amazon.fr/s?k=bebe&i=baby',
            'https://www.amazon.fr/s?k=enfant&i=toys',
            'https://www.amazon.fr/s?k=educatif&i=toys',
            'https://www.amazon.fr/s?k=construction&i=toys',
            'https://www.amazon.fr/s?k=poupee&i=toys',
            
            # Auto & Moto
            'https://www.amazon.fr/s?k=auto&i=automotive',
            'https://www.amazon.fr/s?k=moto&i=automotive',
            'https://www.amazon.fr/s?k=accessoire+auto&i=automotive',
            'https://www.amazon.fr/s?k=pieces+auto&i=automotive',
            'https://www.amazon.fr/s?k=entretien+auto&i=automotive',
            'https://www.amazon.fr/s?k=gps&i=automotive',
            'https://www.amazon.fr/s?k=dashcam&i=automotive',
            'https://www.amazon.fr/s?k=chargeur+auto&i=automotive',
            'https://www.amazon.fr/s?k=housse+siege&i=automotive',
            'https://www.amazon.fr/s?k=tapis+auto&i=automotive',
            
            # Livres & Culture
            'https://www.amazon.fr/s?k=livre&i=stripbooks',
            'https://www.amazon.fr/s?k=roman&i=stripbooks',
            'https://www.amazon.fr/s?k=bd&i=stripbooks',
            'https://www.amazon.fr/s?k=manga&i=stripbooks',
            'https://www.amazon.fr/s?k=dvd&i=dvd',
            'https://www.amazon.fr/s?k=musique&i=popular',
            'https://www.amazon.fr/s?k=cd&i=popular',
            'https://www.amazon.fr/s?k=vinyl&i=popular',
            'https://www.amazon.fr/s?k=instrument&i=musical-instruments',
            'https://www.amazon.fr/s?k=partition&i=musical-instruments',
            
            # Santé & Bien-être
            'https://www.amazon.fr/s?k=sante&i=hpc',
            'https://www.amazon.fr/s?k=complement&i=hpc',
            'https://www.amazon.fr/s?k=vitamine&i=hpc',
            'https://www.amazon.fr/s?k=massage&i=hpc',
            'https://www.amazon.fr/s?k=medical&i=hpc',
            'https://www.amazon.fr/s?k=bien+etre&i=hpc',
            'https://www.amazon.fr/s?k=relaxation&i=hpc',
            'https://www.amazon.fr/s?k=hygiene&i=hpc',
            'https://www.amazon.fr/s?k=pharmacie&i=hpc',
            'https://www.amazon.fr/s?k=orthopedique&i=hpc',
            
            # Animaux
            'https://www.amazon.fr/s?k=chien&i=pets',
            'https://www.amazon.fr/s?k=chat&i=pets',
            'https://www.amazon.fr/s?k=aquarium&i=pets',
            'https://www.amazon.fr/s?k=oiseau&i=pets',
            'https://www.amazon.fr/s?k=rongeur&i=pets',
            'https://www.amazon.fr/s?k=croquette&i=pets',
            'https://www.amazon.fr/s?k=jouet+chien&i=pets',
            'https://www.amazon.fr/s?k=litiere&i=pets',
            'https://www.amazon.fr/s?k=collier&i=pets',
            'https://www.amazon.fr/s?k=transport+animal&i=pets',
            
            # Alimentation & Boissons
            'https://www.amazon.fr/s?k=epicerie&i=grocery',
            'https://www.amazon.fr/s?k=bio&i=grocery',
            'https://www.amazon.fr/s?k=the&i=grocery',
            'https://www.amazon.fr/s?k=cafe&i=grocery',
            'https://www.amazon.fr/s?k=chocolat&i=grocery',
            'https://www.amazon.fr/s?k=snack&i=grocery',
            'https://www.amazon.fr/s?k=complement+alimentaire&i=grocery',
            'https://www.amazon.fr/s?k=boisson&i=grocery',
            'https://www.amazon.fr/s?k=condiment&i=grocery',
            'https://www.amazon.fr/s?k=conserve&i=grocery',
        ]
        
        # Mélanger toutes les catégories pour un parcours aléatoire
        random.shuffle(category_searches)
        
        return category_searches
    
    custom_settings = {
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'DOWNLOAD_DELAY': 4,
        'RANDOMIZE_DOWNLOAD_DELAY': 2,
        'CONCURRENT_REQUESTS': 1,
        'ROBOTSTXT_OBEY': False,
        'COOKIES_ENABLED': True,
        'RETRY_TIMES': 5,
        'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429, 403],
        'CLOSESPIDER_ITEMCOUNT': 5000,
        'DEFAULT_REQUEST_HEADERS': {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8',  # Français pour Amazon France
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
        },
    }
    
    def start_requests(self):
        """Génère les requêtes initiales avec identification du type de page"""
        for url in self.start_urls:
            # Identifier le type de page pour adapter le parsing
            page_type = self.identify_page_type(url)
            
            yield scrapy.Request(
                url=url,
                callback=self.parse_mixed_page,  # Callback unifié
                headers=self.get_random_headers(),
                meta={'page_type': page_type, 'url': url},
                dont_filter=True
            )
    
    def identify_page_type(self, url):
        """Identifie le type de page pour adapter le parsing"""
        if '/s?' in url:
            return 'category_search'
        elif '/gp/goldbox' in url:
            return 'deals'
        else:
            return 'search'
    
    def parse_mixed_page(self, response):
        """Parse unifié pour toutes les pages de recherche par catégorie"""
        if self.products_scraped >= self.target_products:
            return
        
        if response.status in [503, 403] or 'signin' in response.url:
            self.logger.warning(f"Access blocked for {response.url}")
            return
        
        page_type = response.meta.get('page_type', 'unknown')
        source_url = response.meta.get('url', response.url)
        
        self.logger.info(f"Exploring {page_type} page: {source_url}")
        
        # Extraire les liens de produits depuis les pages de recherche
        product_links = self.extract_search_links(response)
        
        if not product_links:
            self.logger.warning(f"No products found on {page_type} page")
            return
        
        self.logger.info(f"Found {len(product_links)} products on {page_type} page")
        
        # Mélanger les produits pour un échantillonnage aléatoire
        random.shuffle(product_links)
        
        # Traiter chaque produit (limité pour éviter de surcharger une catégorie)
        for link in product_links[:20]:  # Prendre max 20 produits par page pour plus de diversité
            if self.products_scraped >= self.target_products:
                break
                
            product_url = urljoin('https://www.amazon.fr', link) if link.startswith('/') else link
            asin = self.extract_asin(product_url)
            
            if asin and asin not in self.processed_asins:
                self.processed_asins.add(asin)
                
                yield scrapy.Request(
                    url=product_url,
                    callback=self.parse_product,
                    meta={'asin': asin, 'source_type': page_type, 'source_url': source_url},
                    headers=self.get_random_headers(),
                    dont_filter=True
                )
        
        # Suivre la pagination de façon aléatoire (pas toutes les pages)
        if self.products_scraped < self.target_products and random.random() < 0.3:  # 30% de chance de suivre la pagination
            next_page = self.get_next_search_page(response)
            if next_page:
                yield scrapy.Request(
                    url=next_page,
                    callback=self.parse_mixed_page,
                    headers=self.get_random_headers(),
                    meta={'page_type': page_type, 'url': source_url},
                    dont_filter=True
                )
    
    def extract_bestseller_links(self, response):
        """Extrait les liens depuis les pages bestsellers"""
        selectors = [
            # Pages bestsellers classiques
            '.zg-grid-general-faceout .a-link-normal::attr(href)',
            '.zg-item-container .a-link-normal::attr(href)',
            '.zg-grid-general-faceout a[href*="/dp/"]::attr(href)',
            '.zg_itemImmersion .a-link-normal::attr(href)',
            
            # Pages new releases
            '.nr-grid-general-faceout .a-link-normal::attr(href)',
            '.nr-item-container .a-link-normal::attr(href)',
            
            # Pages most wished for
            '.mw-grid-general-faceout .a-link-normal::attr(href)',
            '.mw-item-container .a-link-normal::attr(href)',
            
            # Sélecteurs génériques Amazon
            'a[href*="/dp/"]::attr(href)',
            '.s-result-item a[href*="/dp/"]::attr(href)',
        ]
        
        for selector in selectors:
            links = response.css(selector).getall()
            if links:
                clean_links = []
                for link in links:
                    if '/dp/' in link:
                        if '?' in link:
                            link = link.split('?')[0]
                        clean_links.append(link)
                
                if clean_links:
                    return clean_links[:50]
        
        return []
    
    def extract_search_links(self, response):
        """Extrait les liens depuis les pages de recherche (produits mal notés, deals)"""
        selectors = [
            # Résultats de recherche standard
            '[data-component-type="s-search-result"] h2 a::attr(href)',
            '.s-result-item [data-cy="title-recipe-link"]::attr(href)',
            '.s-result-item h2 a::attr(href)',
            '.s-search-result [data-testid="title"]::attr(href)',
            '.s-result-item .a-link-normal::attr(href)',
            
            # Pages deals/goldbox
            '.deal-container a[href*="/dp/"]::attr(href)',
            '.gb-deal-link::attr(href)',
            '.deal-item a[href*="/dp/"]::attr(href)',
            
            # Sélecteurs génériques
            'a[href*="/dp/"]::attr(href)',
        ]
        
        for selector in selectors:
            links = response.css(selector).getall()
            if links:
                clean_links = []
                for link in links:
                    if '/dp/' in link:
                        if '?' in link:
                            link = link.split('?')[0]
                        clean_links.append(link)
                
                if clean_links:
                    return clean_links[:30]
        
        return []
    
    def get_next_page_url(self, response, page_type):
        """Récupère la page suivante selon le type de page"""
        if page_type in ['bestseller', 'new_releases', 'wished_for']:
            return self.get_next_bestseller_page(response)
        elif page_type in ['low_rated_search', 'search', 'deals']:
            return self.get_next_search_page(response)
        else:
            next_page = self.get_next_bestseller_page(response)
            if not next_page:
                next_page = self.get_next_search_page(response)
            return next_page
    
    def get_next_bestseller_page(self, response):
        """Récupère la page suivante pour les bestsellers"""
        next_selectors = [
            '.a-pagination .a-last a::attr(href)',
            'a[aria-label="Next page"]::attr(href)',
            '.a-pagination .a-normal:last-child a::attr(href)',
        ]
        
        for selector in next_selectors:
            next_link = response.css(selector).get()
            if next_link:
                return urljoin(response.url, next_link)
        
        return None
    
    def get_next_search_page(self, response):
        """Récupère la page suivante pour les recherches"""
        next_selectors = [
            '.a-pagination .a-last a::attr(href)',
            'a[aria-label="Go to next page"]::attr(href)',
            '.s-pagination-next::attr(href)',
            'a[aria-label="Next"]::attr(href)',
        ]
        
        for selector in next_selectors:
            next_link = response.css(selector).get()
            if next_link:
                return urljoin(response.url, next_link)
        
        return None
    
    def parse_product(self, response):
        """Parse une page produit"""
        if response.status in [503, 403, 404] or 'signin' in response.url:
            return
        
        asin = response.meta.get('asin')
        source_type = response.meta.get('source_type', 'unknown')
        source_url = response.meta.get('source_url', '')
        
        if not asin:
            return
        
        # Extraire les informations du produit
        product_title = response.css('#productTitle::text').get()
        if product_title:
            product_title = product_title.strip()
        else:
            product_title = f"Product {asin}"
        
        # Extraire la note moyenne du produit pour classification
        avg_rating = self.extract_product_rating(response)
        rating_category = self.classify_rating(avg_rating)
        
        self.products_scraped += 1
        self.logger.info(f"Product {self.products_scraped}/1000: {product_title} (ASIN: {asin}) [Category: {source_type}] [Rating: {avg_rating}★ - {rating_category}]")
        
        # Extraire les commentaires
        reviews_found = 0
        review_selectors = [
            '[data-hook="review"]',
            '.review',
            '[data-testid="review"]',
            '.cr-widget-Reviews [data-hook="review"]',
            '.reviewText',
        ]
        
        reviews = []
        for selector in review_selectors:
            reviews = response.css(selector)
            if reviews:
                break
        
        # Traiter chaque commentaire
        for review in reviews:
            review_data = self.extract_review_data(review, asin, response.url, product_title, source_type, rating_category)
            if review_data:
                reviews_found += 1
                yield review_data
        
        # Essayer d'obtenir plus de commentaires si peu trouvés
        if reviews_found < 3:
            reviews_link = response.css('a[data-hook="see-all-reviews-link-foot"]::attr(href)').get()
            if not reviews_link:
                reviews_link = response.css(f'a[href*="/product-reviews/{asin}"]::attr(href)').get()
            
            if reviews_link:
                reviews_url = urljoin(response.url, reviews_link)
                yield scrapy.Request(
                    url=reviews_url,
                    callback=self.parse_reviews_page,
                    meta={'asin': asin, 'product_title': product_title, 'source_type': source_type, 'rating_category': rating_category},
                    headers=self.get_random_headers()
                )
        
        self.logger.info(f"Found {reviews_found} direct reviews for {asin} ({rating_category} product)")
    
    def extract_product_rating(self, response):
        """Extrait la note moyenne du produit"""
        rating_selectors = [
            '.a-icon-alt::text',
            '[data-hook="average-star-rating"] .a-icon-alt::text',
            '.arp-rating-out-of-text::text',
            '.a-star-medium .a-icon-alt::text',
        ]
        
        for selector in rating_selectors:
            rating_text = response.css(selector).get()
            if rating_text and ('sur' in rating_text.lower() or 'out of' in rating_text.lower()):
                rating_match = re.search(r'(\d+\.?\d*)', rating_text)
                if rating_match:
                    return float(rating_match.group(1))
        
        return None
    
    def classify_rating(self, rating):
        """Classifie la note en catégorie"""
        if rating is None:
            return 'unrated'
        elif rating >= 4.5:
            return 'excellent'
        elif rating >= 4.0:
            return 'good'
        elif rating >= 3.0:
            return 'average'
        elif rating >= 2.0:
            return 'poor'
        else:
            return 'very_poor'
    
    def parse_reviews_page(self, response):
        """Parse une page dédiée aux commentaires"""
        if response.status in [503, 403] or 'signin' in response.url:
            self.logger.warning(f"Reviews page blocked: {response.url}")
            return
        
        asin = response.meta.get('asin')
        product_title = response.meta.get('product_title')
        source_type = response.meta.get('source_type')
        rating_category = response.meta.get('rating_category')
        
        reviews = response.css('[data-hook="review"]')
        reviews_found = 0
        
        for review in reviews:
            review_data = self.extract_review_data(review, asin, response.url, product_title, source_type, rating_category)
            if review_data:
                reviews_found += 1
                yield review_data
        
        self.logger.info(f"Found {reviews_found} additional reviews for {asin} ({rating_category} product)")
    
    def extract_review_data(self, review, asin, product_url, product_title, source_type, rating_category):
        """Extrait les données d'un commentaire avec classification"""
        try:
            # Nom du reviewer
            reviewer_name = review.css('.a-profile-name::text').get()
            if not reviewer_name:
                reviewer_name = review.css('[data-hook="review-author"] .a-profile-name::text').get()
            
            # Note du commentaire
            rating_text = review.css('.a-icon-alt::text').get()
            rating = self.extract_rating(rating_text)
            
            # Classification du commentaire
            review_sentiment = self.classify_review_rating(rating)
            
            # Titre du commentaire
            title = review.css('[data-hook="review-title"] span::text').get()
            if not title:
                title = review.css('.review-title::text').get()
            
            # Date
            date_text = review.css('[data-hook="review-date"]::text').get()
            
            # Texte du commentaire
            review_text_parts = review.css('[data-hook="review-body"] span::text').getall()
            review_text = ' '.join(review_text_parts).strip() if review_text_parts else ""
            
            if not review_text:
                review_text_parts = review.css('.reviewText span::text').getall()
                review_text = ' '.join(review_text_parts).strip()
            
            # Votes utiles
            helpful_votes_text = review.css('[data-hook="helpful-vote-statement"]::text').get()
            helpful_votes = self.extract_helpful_votes(helpful_votes_text)
            
            # Achat vérifié
            verified_purchase = bool(review.css('[data-hook="avp-badge"]'))
            
            # Retourner les données enrichies
            if title or review_text:
                return {
                    'asin': asin,
                    'product_title': product_title,
                    'product_url': product_url,
                    'product_rating_category': rating_category,  # excellent/good/poor/etc
                    'source_type': source_type,  # bestseller/low_rated_search/deals/etc
                    'reviewer_name': reviewer_name or 'Anonymous',
                    'review_rating': rating,
                    'review_sentiment': review_sentiment,  # positive/negative/neutral
                    'title': title or '',
                    'date': self.clean_date(date_text),
                    'verified_purchase': verified_purchase,
                    'review_text': review_text,
                    'helpful_votes': helpful_votes,
                }
        except Exception as e:
            self.logger.error(f"Error extracting review: {e}")
        
        return None
    
    def classify_review_rating(self, rating):
        """Classifie un commentaire selon sa note"""
        if rating is None:
            return 'unrated'
        elif rating >= 4:
            return 'positive'
        elif rating >= 3:
            return 'neutral'
        else:
            return 'negative'
    
    def get_random_headers(self):
        """Headers aléatoires pour éviter la détection"""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
        ]
        
        return {
            'User-Agent': random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8',  # Français pour Amazon France
            'Referer': 'https://www.amazon.fr/',
        }
    
    def extract_asin(self, url):
        """Extrait l'ASIN depuis l'URL"""
        asin_match = re.search(r'/dp/([A-Z0-9]{10})', url)
        if asin_match:
            return asin_match.group(1)
        return None
    
    def extract_rating(self, rating_text):
        """Extrait la note numérique"""
        if rating_text:
            rating_match = re.search(r'(\d+\.?\d*)', rating_text)
            if rating_match:
                return float(rating_match.group(1))
        return None
    
    def extract_helpful_votes(self, helpful_text):
        """Extrait le nombre de votes utiles"""
        if helpful_text:
            votes_match = re.search(r'(\d+)', helpful_text)
            if votes_match:
                return int(votes_match.group(1))
        return 0
    
    def clean_date(self, date_text):
        """Nettoie la date"""
        if date_text:
            cleaned = re.sub(r'^[^0-9]*', '', date_text.strip())
            return cleaned
        return None


# Pipeline de traitement avec analyse de sentiment
class AmazonReviewsPipeline:
    def __init__(self):
        self.items = []
        self.products_count = 0
        self.reviews_per_product = {}
        self.source_type_stats = {}
        self.sentiment_stats = {'positive': 0, 'negative': 0, 'neutral': 0, 'unrated': 0}
        self.rating_category_stats = {}
    
    def process_item(self, item, spider):
        self.items.append(item)
        
        # Statistiques par produit
        asin = item['asin']
        if asin not in self.reviews_per_product:
            self.reviews_per_product[asin] = 0
            self.products_count += 1
        self.reviews_per_product[asin] += 1
        
        # Statistiques par type de source
        source_type = item.get('source_type', 'unknown')
        if source_type not in self.source_type_stats:
            self.source_type_stats[source_type] = {'products': set(), 'reviews': 0}
        self.source_type_stats[source_type]['products'].add(asin)
        self.source_type_stats[source_type]['reviews'] += 1
        
        # Statistiques par sentiment
        sentiment = item.get('review_sentiment', 'unrated')
        if sentiment in self.sentiment_stats:
            self.sentiment_stats[sentiment] += 1
        
        # Statistiques par catégorie de note produit
        rating_cat = item.get('product_rating_category', 'unrated')
        if rating_cat not in self.rating_category_stats:
            self.rating_category_stats[rating_cat] = 0
        self.rating_category_stats[rating_cat] += 1
        
        return item
    
    def close_spider(self, spider):
        import csv
        from datetime import datetime
        import unicodedata
        import html
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f'amazon_france_random_reviews_{timestamp}.csv'
        
        if self.items:
            with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
                fieldnames = [
                    'asin', 'product_title', 'product_url', 'product_rating_category', 
                    'source_type', 'reviewer_name', 'review_rating', 'review_sentiment',
                    'title', 'date', 'verified_purchase', 'review_text', 'helpful_votes'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for item in self.items:
                    clean_item = {}
                    for field in fieldnames:
                        value = item.get(field, '')
                        
                        if isinstance(value, str):
                            # Nettoyage Unicode et caractères spéciaux
                            value = html.unescape(value)
                            value = unicodedata.normalize('NFKC', value)
                            
                            # Corrections courantes pour le français
                            replacements = {
                                'â€™': "'", 'â€œ': '"', 'â€': '"', 'â€"': '—',
                                'â€¦': '...',
                            }
                            
                            for wrong, correct in replacements.items():
                                value = value.replace(wrong, correct)
                            
                            # Nettoyage final
                            value = value.replace('\n', ' ').replace('\r', ' ')
                            value = value.replace('"', '""')
                            value = re.sub(r'\s+', ' ', value).strip()
                            
                        elif value is None:
                            value = ''
                        
                        clean_item[field] = value
                    
                    writer.writerow(clean_item)
        
        # Sauvegarde JSON
        json_filename = f'amazon_france_random_reviews_{timestamp}.json'
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(self.items, f, ensure_ascii=False, indent=2)
        
        # Rapport final détaillé avec analyse de sentiment
        total_reviews = len(self.items)
        spider.logger.info(f"=== FINAL REPORT - AMAZON FRANCE RANDOM CATEGORY REVIEWS ===")
        spider.logger.info(f"Unique products processed: {self.products_count}")
        spider.logger.info(f"Total reviews collected: {total_reviews}")
        spider.logger.info(f"Progress: {spider.products_scraped}/1000 products explored")
        spider.logger.info(f"Average reviews/product: {total_reviews/self.products_count:.1f}" if self.products_count > 0 else "0")
        
        # Analyse de sentiment
        spider.logger.info(f"=== SENTIMENT ANALYSIS ===")
        total_rated = sum(self.sentiment_stats.values())
        for sentiment, count in self.sentiment_stats.items():
            percentage = (count/total_rated)*100 if total_rated > 0 else 0
            spider.logger.info(f"  {sentiment.capitalize()} reviews: {count} ({percentage:.1f}%)")
        
        # Top produits par sentiment
        if self.reviews_per_product:
            top_products = sorted(self.reviews_per_product.items(), key=lambda x: x[1], reverse=True)[:10]
            spider.logger.info(f"=== TOP 10 MOST REVIEWED PRODUCTS ===")
            for asin, count in top_products:
                spider.logger.info(f"  ASIN {asin}: {count} reviews")
        
        # Statistiques par type de source
        spider.logger.info(f"=== DISTRIBUTION BY CATEGORY ===")
        for source_type, stats in self.source_type_stats.items():
            products_count = len(stats['products'])
            reviews_count = stats['reviews']
            spider.logger.info(f"  {source_type}: {products_count} products, {reviews_count} reviews")
        
        # Statistiques par catégorie de note produit
        spider.logger.info(f"=== DISTRIBUTION BY PRODUCT RATING ===")
        for rating_cat, count in self.rating_category_stats.items():
            percentage = (count/total_reviews)*100 if total_reviews > 0 else 0
            spider.logger.info(f"  {rating_cat} products: {count} reviews ({percentage:.1f}%)")
        
        spider.logger.info(f"=== FILES GENERATED ===")
        spider.logger.info(f"  - CSV: {csv_filename}")
        spider.logger.info(f"  - JSON: {json_filename}")
        spider.logger.info(f"=== RANDOM SAMPLING ACHIEVED ===")
        spider.logger.info(f"  ✓ Electronics & High-Tech")
        spider.logger.info(f"  ✓ Maison & Jardin")
        spider.logger.info(f"  ✓ Mode & Beauté")
        spider.logger.info(f"  ✓ Sport & Loisirs")
        spider.logger.info(f"  ✓ Jouets & Jeux")
        spider.logger.info(f"  ✓ Auto & Moto")
        spider.logger.info(f"  ✓ Livres & Culture")
        spider.logger.info(f"  ✓ Santé & Bien-être")
        spider.logger.info(f"  ✓ Animaux")
        spider.logger.info(f"  ✓ Alimentation & Boissons")
        spider.logger.info(f"  ✓ Random products from all categories!")


# Configuration finale
SETTINGS = {
    'BOT_NAME': 'amazon_france_random_reviews',
    'SPIDER_MODULES': ['__main__'],
    'NEWSPIDER_MODULE': '__main__',
    'ROBOTSTXT_OBEY': False,
    'USER_AGENT': 'amazon_france_random_reviews (+http://www.yourdomain.com)',
    'DOWNLOAD_DELAY': 4,
    'RANDOMIZE_DOWNLOAD_DELAY': 2,
    'CONCURRENT_REQUESTS': 1,
    'CLOSESPIDER_ITEMCOUNT': 5000,
    'ITEM_PIPELINES': {
        '__main__.AmazonReviewsPipeline': 300,
    },
    'AUTOTHROTTLE_ENABLED': True,
    'AUTOTHROTTLE_START_DELAY': 3,
    'AUTOTHROTTLE_MAX_DELAY': 15,
    'AUTOTHROTTLE_TARGET_CONCURRENCY': 0.5,
}


if __name__ == "__main__":
    from scrapy.crawler import CrawlerProcess
    
    process = CrawlerProcess(SETTINGS)
    process.crawl(AmazonReviewsSpider)
    process.start()
