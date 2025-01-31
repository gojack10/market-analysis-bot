# Use Python 3.13 RC slim image as base
FROM python:3.13-rc-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Add a default value for ALPHA_VANTAGE_API_KEY that will be overridden at runtime
ENV ALPHA_VANTAGE_API_KEY="default_value_replace_at_runtime"

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (including autotools-dev for updated config.guess/sub)
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    autotools-dev \
    && rm -rf /var/lib/apt/lists/*

# Build & install TA-Lib from source, copying modern config.guess/config.sub
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xvzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && cp /usr/share/misc/config.guess . \
    && cp /usr/share/misc/config.sub . \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib-0.4.0-src.tar.gz ta-lib/

# Create symlink and update linker cache
RUN if [ -f /usr/lib/libta_lib.so ]; then \
      ln -s /usr/lib/libta_lib.so /usr/lib/libta-lib.so; \
    fi \
    && ldconfig

# Set TA-Lib environment variables for Python wrapper
ENV TA_LIBRARY_PATH=/usr/lib \
    TA_INCLUDE_PATH=/usr/include

# Copy only requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files into the container
COPY . .

# Command to run your Python script
CMD ["python", "analysis_bot.py"] 